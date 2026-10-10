"""Multi-hop query decomposition and execution for complex reasoning queries."""

from __future__ import annotations

import re
import time

from pydantic import BaseModel, Field

from signalrag.models.retrieval import SearchResult
from signalrag.retrieval.pipeline import RetrievalPipeline


class SubQuery(BaseModel):
    """An individual atomic question extracted from a complex parent query."""

    id: str
    query: str
    intent: str = "fact"
    depends_on: list[str] = Field(default_factory=list)


class QueryDecompositionPlan(BaseModel):
    """Execution plan containing decomposed sub-queries and execution mode."""

    original_query: str
    sub_queries: list[SubQuery] = Field(default_factory=list)
    is_complex: bool = True
    execution_strategy: str = "parallel"  # parallel or sequential


class SubQueryResult(BaseModel):
    """Retrieval outcome for an individual sub-query."""

    sub_query: SubQuery
    results: list[SearchResult] = Field(default_factory=list)
    latency_ms: float = 0.0


class MultiHopTrace(BaseModel):
    """Diagnostic trace of multi-hop query decomposition and execution."""

    original_query: str
    plan: QueryDecompositionPlan
    sub_results: list[SubQueryResult] = Field(default_factory=list)
    merged_results: list[SearchResult] = Field(default_factory=list)
    total_latency_ms: float = 0.0


class QueryDecomposer:
    """Decomposes compound, comparative, and multi-part queries into atomic sub-queries."""

    SPLIT_PATTERNS = [
        re.compile(r"\s+(?:and|as well as|additionally)\s+", re.IGNORECASE),
        re.compile(r"\s+(?:compared to|versus|vs\.?)\s+", re.IGNORECASE),
        re.compile(r"\s+(?:while|whereas|meanwhile)\s+", re.IGNORECASE),
        re.compile(r";|\?+\s*", re.IGNORECASE),
    ]

    def decompose(self, query: str) -> QueryDecompositionPlan:
        """Analyze query and produce an atomic sub-query decomposition plan."""
        cleaned = query.strip()
        parts: list[str] = []

        # Check for comparative queries
        if re.search(r"\b(compare|difference between|versus|vs\.?)\b", cleaned, re.IGNORECASE):
            parts = self._split_comparative(cleaned)
        else:
            parts = self._split_compound(cleaned)

        # Filter and sanitize sub-queries
        sub_queries: list[SubQuery] = []
        for idx, part in enumerate(parts):
            p = part.strip().rstrip("?.")
            if not p:
                continue
            # Ensure question form
            sub_q_text = p if p.endswith("?") else f"{p}?"
            intent = "comparative" if "compare" in cleaned.lower() else "factual"
            sub_queries.append(
                SubQuery(
                    id=f"sub_{idx + 1}",
                    query=sub_q_text,
                    intent=intent,
                )
            )

        if len(sub_queries) <= 1:
            return QueryDecompositionPlan(
                original_query=query,
                sub_queries=[SubQuery(id="sub_1", query=query, intent="factual")],
                is_complex=False,
                execution_strategy="direct",
            )

        return QueryDecompositionPlan(
            original_query=query,
            sub_queries=sub_queries,
            is_complex=True,
            execution_strategy="parallel",
        )

    def _split_comparative(self, query: str) -> list[str]:
        # Handle "Compare X and Y" or "Difference between X and Y"
        match = re.search(
            r"(?:compare|difference between)\s+(.*?)(?:\s+(?:and|versus|vs\.?)\s+)(.*)",
            query,
            re.IGNORECASE,
        )
        if match:
            item_a = match.group(1).strip()
            item_b = match.group(2).strip().rstrip("?.")
            return [f"What are the properties of {item_a}", f"What are the properties of {item_b}"]

        return self._split_compound(query)

    def _split_compound(self, query: str) -> list[str]:
        for pattern in self.SPLIT_PATTERNS:
            parts = pattern.split(query)
            if len(parts) > 1 and all(len(p.strip()) > 5 for p in parts):
                return parts
        return [query]


class MultiHopRetriever:
    """Executes decomposed sub-queries across retrieval pipeline and merges candidate evidence."""

    def __init__(self, pipeline: RetrievalPipeline, decomposer: QueryDecomposer | None = None):
        self.pipeline = pipeline
        self.decomposer = decomposer or QueryDecomposer()

    def retrieve(self, query: str, top_k_per_subquery: int = 3) -> MultiHopTrace:
        """Decompose query and retrieve candidates across all sub-queries."""
        t0 = time.perf_counter()
        plan = self.decomposer.decompose(query)

        sub_results: list[SubQueryResult] = []
        seen_chunk_ids: set[str] = set()
        merged_results: list[SearchResult] = []

        for sub_q in plan.sub_queries:
            t_sub0 = time.perf_counter()
            res = self.pipeline.retrieve(sub_q.query, top_k=top_k_per_subquery)
            sub_lat = (time.perf_counter() - t_sub0) * 1000

            sub_results.append(
                SubQueryResult(
                    sub_query=sub_q,
                    results=res,
                    latency_ms=round(sub_lat, 2),
                )
            )

            # Deduplicate and merge candidates
            for r in res:
                if r.chunk.id not in seen_chunk_ids:
                    seen_chunk_ids.add(r.chunk.id)
                    merged_results.append(r)

        total_lat = (time.perf_counter() - t0) * 1000

        return MultiHopTrace(
            original_query=query,
            plan=plan,
            sub_results=sub_results,
            merged_results=merged_results,
            total_latency_ms=round(total_lat, 2),
        )
