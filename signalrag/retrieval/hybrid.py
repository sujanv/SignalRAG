"""Hybrid retriever combining dense semantic search and sparse BM25 search."""

from typing import Any, Literal

from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult
from signalrag.retrieval.base import BaseRetriever
from signalrag.retrieval.bm25 import BM25Retriever
from signalrag.retrieval.semantic import SemanticRetriever


class HybridRetriever(BaseRetriever):
    """Combines vector search and BM25 lexical search using convex score fusion or Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.5,
        fusion_method: Literal["linear", "rrf"] = "linear",
        rrf_k: int = 60,
        candidate_multiplier: int = 2,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be between 0.0 and 1.0, got {alpha}")
        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """Compute RRF scores: RRF_score = sum(weight / (k + rank))."""
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        vec_weight = self.alpha
        bm25_weight = 1.0 - self.alpha

        for rank, res in enumerate(vector_results, start=1):
            cid = res.chunk.id
            chunk_map[cid] = res.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (vec_weight / (self.rrf_k + rank))

        for rank, res in enumerate(bm25_results, start=1):
            cid = res.chunk.id
            chunk_map[cid] = res.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (bm25_weight / (self.rrf_k + rank))

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        fused: list[SearchResult] = []
        for rank, (cid, score) in enumerate(sorted_items, start=1):
            fused.append(
                SearchResult(
                    chunk=chunk_map[cid],
                    score=float(score),
                    retrieval_method="hybrid",
                    rank=rank,
                    metadata_filters_applied={k: str(v) for k, v in (filters or {}).items()},
                )
            )
        return fused

    def _linear_score_fusion(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """Linear weighted combination of normalized vector and BM25 scores: score = alpha*vec + (1-alpha)*bm25."""
        vec_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for res in vector_results:
            cid = res.chunk.id
            chunk_map[cid] = res.chunk
            vec_scores[cid] = res.score

        for res in bm25_results:
            cid = res.chunk.id
            chunk_map[cid] = res.chunk
            bm25_scores[cid] = res.score

        all_ids = set(vec_scores.keys()).union(set(bm25_scores.keys()))
        combined_scores: dict[str, float] = {}

        for cid in all_ids:
            v_score = vec_scores.get(cid, 0.0)
            b_score = bm25_scores.get(cid, 0.0)
            combined_scores[cid] = (self.alpha * v_score) + ((1.0 - self.alpha) * b_score)

        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        fused: list[SearchResult] = []
        for rank, (cid, score) in enumerate(sorted_items, start=1):
            fused.append(
                SearchResult(
                    chunk=chunk_map[cid],
                    score=float(score),
                    retrieval_method="hybrid",
                    rank=rank,
                    metadata_filters_applied={k: str(v) for k, v in (filters or {}).items()},
                )
            )
        return fused

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Execute hybrid search combining both semantic and BM25 retrievers."""
        if not query.strip() or top_k <= 0:
            return []

        # If alpha is 1.0, purely vector
        if self.alpha == 1.0:
            return self.semantic_retriever.retrieve(query, top_k=top_k, filters=filters)

        # If alpha is 0.0, purely BM25
        if self.alpha == 0.0:
            return self.bm25_retriever.retrieve(query, top_k=top_k, filters=filters)

        candidate_k = top_k * self.candidate_multiplier

        vec_results = self.semantic_retriever.retrieve(query, top_k=candidate_k, filters=filters)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=candidate_k, filters=filters)

        if self.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(vec_results, bm25_results, top_k, filters)
        else:
            return self._linear_score_fusion(vec_results, bm25_results, top_k, filters)
