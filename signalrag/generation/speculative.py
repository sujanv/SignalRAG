"""Speculative RAG pipeline producing low-latency draft answers and verifying citations."""

from __future__ import annotations

import re
import time

from pydantic import BaseModel, Field

from signalrag.models.chunk import Chunk


class SpeculativeDraft(BaseModel):
    """Initial low-latency hypothesis draft answer and proposed citations."""

    draft_answer: str
    proposed_chunk_ids: list[str] = Field(default_factory=list)
    draft_latency_ms: float = 0.0


class VerifiedSpeculativeResponse(BaseModel):
    """Verified and refined response with draft verification telemetry."""

    query: str
    draft: SpeculativeDraft
    verified_answer: str
    modifications_made: bool
    verification_latency_ms: float
    total_latency_ms: float
    grounding_confidence: float


class SpeculativeRAGEngine:
    """Implements Speculative RAG: generates rapid draft predictions and validates claims against ground evidence."""

    def draft(self, query: str, candidate_chunks: list[Chunk]) -> SpeculativeDraft:
        """Produce rapid speculative draft using salient chunk sentences."""
        t0 = time.perf_counter()
        if not candidate_chunks:
            return SpeculativeDraft(
                draft_answer="Insufficient context available to answer the query.",
                proposed_chunk_ids=[],
                draft_latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        # Extract top sentence from first chunk as draft hypothesis
        first_chunk = candidate_chunks[0]
        sentences = [s.strip() for s in re.split(r"[.!?]", first_chunk.text) if len(s.strip()) > 15]
        core_sentence = sentences[0] if sentences else first_chunk.text[:120]
        draft_text = f"Based on preliminary retrieval: {core_sentence} [1]."

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpeculativeDraft(
            draft_answer=draft_text,
            proposed_chunk_ids=[first_chunk.id],
            draft_latency_ms=elapsed,
        )

    def verify(
        self,
        query: str,
        draft: SpeculativeDraft,
        ground_chunks: list[Chunk],
    ) -> VerifiedSpeculativeResponse:
        """Verify draft assertions against retrieved ground truth and refine unsupported statements."""
        t0 = time.perf_counter()
        if not ground_chunks:
            return VerifiedSpeculativeResponse(
                query=query,
                draft=draft,
                verified_answer="Refusal: No verified documentation found.",
                modifications_made=True,
                verification_latency_ms=0.1,
                total_latency_ms=draft.draft_latency_ms + 0.1,
                grounding_confidence=0.0,
            )

        # Verify whether draft keywords appear in ground chunks
        draft_words = set(re.findall(r"\w+", draft.draft_answer.lower()))
        ground_text = " ".join(c.text.lower() for c in ground_chunks)
        ground_words = set(re.findall(r"\w+", ground_text))

        overlap = len(draft_words & ground_words) / max(1, len(draft_words))
        modifications_made = False

        if overlap >= 0.4:
            # Verified draft is well-grounded
            verified_text = draft.draft_answer
            confidence = round(min(0.99, overlap * 1.3), 2)
        else:
            # Refine draft with direct ground chunk evidence
            verified_text = f"Verified: {ground_chunks[0].text[:140]}... [1]"
            modifications_made = True
            confidence = 0.85

        ver_elapsed = round((time.perf_counter() - t0) * 1000, 2)
        total_ms = round(draft.draft_latency_ms + ver_elapsed, 2)

        return VerifiedSpeculativeResponse(
            query=query,
            draft=draft,
            verified_answer=verified_text,
            modifications_made=modifications_made,
            verification_latency_ms=ver_elapsed,
            total_latency_ms=total_ms,
            grounding_confidence=confidence,
        )
