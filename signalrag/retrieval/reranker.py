"""Reranking subsystem for re-scoring and re-ordering retrieved candidate chunks."""

import math
import re
from abc import ABC, abstractmethod
from typing import Any

from signalrag.models.retrieval import SearchResult


class BaseReranker(ABC):
    """Abstract interface for second-stage rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Score and re-rank candidate search results."""
        pass


class HeuristicCrossReranker(BaseReranker):
    """Deterministic cross-feature reranker evaluating joint term coverage, phrase matching, and section relevance."""

    def __init__(
        self,
        phrase_weight: float = 0.4,
        coverage_weight: float = 0.4,
        section_weight: float = 0.2,
    ) -> None:
        self.phrase_weight = phrase_weight
        self.coverage_weight = coverage_weight
        self.section_weight = section_weight

    def _score_candidate(self, query_terms: list[str], raw_query: str, res: SearchResult) -> float:
        """Compute cross-feature relevance score."""
        text = res.chunk.text.lower()
        section = (res.chunk.metadata.section_title or "").lower()

        # 1. Exact phrase match
        clean_q = raw_query.lower().strip()
        phrase_score = 1.0 if clean_q in text else 0.0

        # 2. Query terms coverage ratio
        matched_terms = sum(1 for t in query_terms if t in text)
        coverage_score = matched_terms / max(1, len(query_terms))

        # 3. Section heading relevance
        section_score = 0.0
        if section:
            section_matches = sum(1 for t in query_terms if t in section)
            section_score = section_matches / max(1, len(query_terms))

        # 4. First-stage base score integration (scaled)
        base_score = max(0.0, min(1.0, res.score))

        # Combined rerank score in [0.0, 1.0]
        score = (
            (self.phrase_weight * phrase_score)
            + (self.coverage_weight * coverage_score)
            + (self.section_weight * section_score)
        )
        # Blend slightly with base score
        final_score = (0.7 * score) + (0.3 * base_score)
        return round(final_score, 4)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 3,
    ) -> list[SearchResult]:
        if not results or top_k <= 0 or not query.strip():
            return results[:top_k]

        query_terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
        if not query_terms:
            query_terms = query.lower().split()

        scored_results: list[SearchResult] = []
        for res in results:
            new_res = res.model_copy(deep=True)
            rerank_score = self._score_candidate(query_terms, query, new_res)
            new_res.rerank_score = rerank_score
            new_res.score = rerank_score
            new_res.retrieval_method = "reranked"
            scored_results.append(new_res)

        # Sort descending by rerank_score
        scored_results.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)

        # Re-assign ranks
        trimmed = scored_results[:top_k]
        for idx, r in enumerate(trimmed, start=1):
            r.rank = idx

        return trimmed


class CrossEncoderReranker(BaseReranker):
    """Second-stage neural cross-encoder using sentence_transformers.CrossEncoder with heuristic fallback."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._fallback = HeuristicCrossReranker()

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception:
                self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 3,
    ) -> list[SearchResult]:
        if not results or top_k <= 0 or not query.strip():
            return results[:top_k]

        model = self._load_model()
        if model is None:
            return self._fallback.rerank(query, results, top_k=top_k)

        pairs = [[query, r.chunk.text] for r in results]
        raw_scores = model.predict(pairs)

        scored_results: list[SearchResult] = []
        for res, raw_s in zip(results, raw_scores, strict=False):
            new_res = res.model_copy(deep=True)
            # Sigmoidal score scaling
            norm_score = float(1.0 / (1.0 + math.exp(-raw_s)))
            new_res.rerank_score = round(norm_score, 4)
            new_res.score = round(norm_score, 4)
            new_res.retrieval_method = "reranked"
            scored_results.append(new_res)

        scored_results.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)
        trimmed = scored_results[:top_k]
        for idx, r in enumerate(trimmed, start=1):
            r.rank = idx

        return trimmed
