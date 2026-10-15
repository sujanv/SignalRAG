"""Dynamic query-conditioned hybrid fusion with normalized score interpolation."""

from __future__ import annotations

import re

from pydantic import BaseModel

from signalrag.models.retrieval import SearchResult


class FusionWeights(BaseModel):
    """Dynamic retrieval weights computed for a query."""

    vector_weight: float
    bm25_weight: float
    query_type: str


class AdaptiveHybridFuser:
    """Dynamically weights dense semantic vs sparse lexical signals based on query semantics."""

    KEYWORD_REGEX = re.compile(r"\b[A-Z0-9_]{3,}\b|['\"][^'\"]+['\"]|\d{3,}")
    CONCEPTUAL_REGEX = re.compile(
        r"\b(why|how|explain|concept|overview|describe|relationship|impact)\b", re.IGNORECASE
    )

    def determine_weights(self, query: str) -> FusionWeights:
        """Compute query-specific fusion weights alpha (vector) and 1 - alpha (BM25)."""
        q = query.strip()
        words = q.split()

        # Keyword / exact-identifier query
        if self.KEYWORD_REGEX.search(q) and len(words) <= 4:
            return FusionWeights(vector_weight=0.2, bm25_weight=0.8, query_type="keyword")

        # Long conceptual / exploratory query
        if self.CONCEPTUAL_REGEX.search(q) or len(words) >= 8:
            return FusionWeights(vector_weight=0.8, bm25_weight=0.2, query_type="conceptual")

        # Default balanced hybrid
        return FusionWeights(vector_weight=0.5, bm25_weight=0.5, query_type="balanced")

    def normalize_scores(self, results: list[SearchResult]) -> dict[str, float]:
        """Normalize candidate scores to [0.0, 1.0] range via min-max scaling."""
        if not results:
            return {}
        scores = [r.score for r in results]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return {r.chunk.id: 1.0 for r in results}
        return {r.chunk.id: (r.score - min_s) / (max_s - min_s) for r in results}

    def fuse(
        self,
        query: str,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Blend vector and BM25 candidates according to dynamic query weighting."""
        weights = self.determine_weights(query)

        norm_vec = self.normalize_scores(vector_results)
        norm_bm25 = self.normalize_scores(bm25_results)

        chunk_map = {r.chunk.id: r.chunk for r in vector_results + bm25_results}
        all_ids = set(chunk_map.keys())

        fused_candidates: list[tuple[str, float]] = []
        for cid in all_ids:
            v_score = norm_vec.get(cid, 0.0)
            b_score = norm_bm25.get(cid, 0.0)
            combined = (weights.vector_weight * v_score) + (weights.bm25_weight * b_score)
            fused_candidates.append((cid, combined))

        fused_candidates.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(chunk=chunk_map[cid], score=round(score, 4))
            for cid, score in fused_candidates[:top_k]
        ]
