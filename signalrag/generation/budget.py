"""Dynamic context window token budget manager for RAG synthesis."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult


def estimate_token_count(text: str) -> int:
    """Fast heuristic token estimator (approx 4 characters per token)."""
    return max(1, len(text) // 4)


class BudgetAllocationResult(BaseModel):
    """Result of token budget allocation across candidate passages."""

    allocated_chunks: list[Chunk] = Field(default_factory=list)
    total_tokens_used: int = 0
    budget_limit: int = 0
    dropped_chunk_count: int = 0
    truncated_last_chunk: bool = False


class TokenBudgetAllocator:
    """Allocates context window tokens intelligently among retrieved passages."""

    def __init__(
        self,
        max_context_tokens: int = 2048,
        system_reserved_tokens: int = 256,
        generation_reserved_tokens: int = 512,
    ):
        self.max_context_tokens = max_context_tokens
        self.system_reserved_tokens = system_reserved_tokens
        self.generation_reserved_tokens = generation_reserved_tokens
        self.available_retrieval_budget = max(
            256, max_context_tokens - system_reserved_tokens - generation_reserved_tokens
        )

    def allocate(
        self,
        candidates: list[SearchResult] | list[Chunk],
        query: str = "",
    ) -> BudgetAllocationResult:
        """Pack candidates into the available token budget, preserving rank order."""
        chunks = [c.chunk if isinstance(c, SearchResult) else c for c in candidates]
        query_tokens = estimate_token_count(query)
        remaining_budget = max(64, self.available_retrieval_budget - query_tokens)

        allocated: list[Chunk] = []
        tokens_used = 0
        dropped = 0
        truncated = False

        for chunk in chunks:
            chunk_tokens = estimate_token_count(chunk.text)
            if tokens_used + chunk_tokens <= remaining_budget:
                allocated.append(chunk)
                tokens_used += chunk_tokens
            else:
                # Check if we can fit partial sentences of this chunk
                avail = remaining_budget - tokens_used
                if avail >= 40:
                    sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
                    fit_sentences = []
                    fit_tokens = 0
                    for s in sentences:
                        st = estimate_token_count(s)
                        if fit_tokens + st <= avail:
                            fit_sentences.append(s)
                            fit_tokens += st
                        else:
                            break

                    if fit_sentences:
                        # Create truncated chunk clone
                        trunc_text = " ".join(fit_sentences)
                        trunc_chunk = Chunk(
                            id=f"{chunk.id}_trunc",
                            document_id=chunk.document_id,
                            text=trunc_text,
                            metadata=chunk.metadata,
                        )
                        allocated.append(trunc_chunk)
                        tokens_used += fit_tokens
                        truncated = True
                dropped += 1

        return BudgetAllocationResult(
            allocated_chunks=allocated,
            total_tokens_used=tokens_used,
            budget_limit=self.available_retrieval_budget,
            dropped_chunk_count=dropped,
            truncated_last_chunk=truncated,
        )
