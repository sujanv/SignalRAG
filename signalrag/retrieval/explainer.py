"""Explainable retrieval attribution and token-level saliency diagnostics."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from signalrag.models.chunk import Chunk


class TokenMatch(BaseModel):
    """Token-level match detail."""

    token: str
    in_query: bool
    in_chunk: bool
    saliency_score: float = 1.0


class RetrievalExplanation(BaseModel):
    """Detailed attribution explaining ranking decisions for a query-chunk pair."""

    query: str
    chunk_id: str
    overall_score: float
    lexical_overlap: float
    dense_similarity: float
    matching_terms: list[str] = Field(default_factory=list)
    token_breakdown: list[TokenMatch] = Field(default_factory=list)
    summary: str


class RetrievalExplainer:
    """Explains why a chunk was retrieved and ranked by decomposing dense and lexical match contributions."""

    def __init__(self, dense_weight: float = 0.6, lexical_weight: float = 0.4):
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def explain(
        self,
        query: str,
        chunk: Chunk,
        dense_similarity: float = 0.75,
    ) -> RetrievalExplanation:
        """Produce an explainable attribution report for a query against a candidate chunk."""
        q_tokens = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        c_tokens = set(re.findall(r"\w+", chunk.text.lower()))

        matching = [t for t in q_tokens if t in c_tokens]
        unique_matches = sorted(set(matching))

        lexical_ratio = len(unique_matches) / max(1, len(set(q_tokens))) if q_tokens else 0.0
        overall = round(
            (dense_similarity * self.dense_weight) + (lexical_ratio * self.lexical_weight), 3
        )

        breakdown = [
            TokenMatch(
                token=t,
                in_query=True,
                in_chunk=(t in c_tokens),
                saliency_score=round(1.0 / max(1, len(q_tokens)), 2),
            )
            for t in set(q_tokens)
        ]

        summary = (
            f"Chunk '{chunk.id}' scored {overall:.3f} driven by {len(unique_matches)} exact query term matches "
            f"({', '.join(unique_matches) or 'none'}) and dense vector cosine similarity of {dense_similarity:.2f}."
        )

        return RetrievalExplanation(
            query=query,
            chunk_id=chunk.id,
            overall_score=overall,
            lexical_overlap=round(lexical_ratio, 3),
            dense_similarity=round(dense_similarity, 3),
            matching_terms=unique_matches,
            token_breakdown=breakdown,
            summary=summary,
        )
