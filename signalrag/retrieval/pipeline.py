"""Full end-to-end Retrieval Pipeline connecting rewriting, hybrid search, reranking, and compression."""

import time
from dataclasses import dataclass, field
from typing import Any

from signalrag.core.config import Settings, get_settings
from signalrag.models.retrieval import SearchResult
from signalrag.retrieval.base import BaseRetriever
from signalrag.retrieval.bm25 import BM25Retriever
from signalrag.retrieval.compressor import BaseContextCompressor, SentenceRelevanceCompressor
from signalrag.retrieval.hybrid import HybridRetriever
from signalrag.retrieval.reranker import BaseReranker, HeuristicCrossReranker
from signalrag.retrieval.rewriter import BaseQueryRewriter, HeuristicQueryRewriter
from signalrag.retrieval.semantic import SemanticRetriever


@dataclass
class RetrievalTrace:
    """Diagnostic trace of intermediate outputs across the retrieval pipeline."""

    raw_query: str
    rewritten_queries: list[str] = field(default_factory=list)
    semantic_candidates_count: int = 0
    bm25_candidates_count: int = 0
    hybrid_candidates_count: int = 0
    reranked_candidates_count: int = 0
    final_results_count: int = 0
    stages_latency_ms: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0


@dataclass
class PipelineRetrievalResponse:
    """Complete retrieval response including ranked results and pipeline trace."""

    results: list[SearchResult]
    trace: RetrievalTrace


class RetrievalPipeline(BaseRetriever):
    """End-to-end retrieval pipeline coordinating:
    Query -> Query Rewrite -> Hybrid Search (BM25 + Vector) -> Reranker -> Context Compression -> Final Chunks.
    """

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        bm25_retriever: BM25Retriever,
        query_rewriter: BaseQueryRewriter | None = None,
        reranker: BaseReranker | None = None,
        compressor: BaseContextCompressor | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or get_settings()

        self.query_rewriter = query_rewriter or HeuristicQueryRewriter()
        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever

        self.hybrid_retriever = HybridRetriever(
            semantic_retriever=semantic_retriever,
            bm25_retriever=bm25_retriever,
            alpha=cfg.retrieval.hybrid_alpha,
            fusion_method="linear",
        )

        self.reranker = reranker or HeuristicCrossReranker()
        self.compressor = compressor or SentenceRelevanceCompressor(
            embedding_service=semantic_retriever.embedding_service
        )

        self.use_reranker = cfg.retrieval.reranker_enabled
        self.use_compression = cfg.retrieval.compression_enabled

    def retrieve_with_trace(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        rerank: bool | None = None,
        compress: bool | None = None,
    ) -> PipelineRetrievalResponse:
        """Execute the complete retrieval pipeline and return results with full diagnostic trace."""
        start_pipeline = time.perf_counter()
        trace = RetrievalTrace(raw_query=query)

        # 1. Query Rewrite Stage
        t0 = time.perf_counter()
        rewritten = self.query_rewriter.rewrite(query)
        trace.rewritten_queries = rewritten
        effective_query = rewritten[0] if rewritten else query
        trace.stages_latency_ms["query_rewrite"] = round((time.perf_counter() - t0) * 1000, 2)

        # 2. Hybrid Retrieval Stage (BM25 + Semantic)
        t1 = time.perf_counter()
        candidate_k = top_k * 3
        candidates = self.hybrid_retriever.retrieve(effective_query, top_k=candidate_k, filters=filters)
        trace.hybrid_candidates_count = len(candidates)
        trace.stages_latency_ms["hybrid_retrieval"] = round((time.perf_counter() - t1) * 1000, 2)

        # 3. Reranking Stage
        should_rerank = self.use_reranker if rerank is None else rerank
        t2 = time.perf_counter()
        if should_rerank and candidates:
            candidates = self.reranker.rerank(effective_query, candidates, top_k=top_k)
            trace.reranked_candidates_count = len(candidates)
        else:
            candidates = candidates[:top_k]
        trace.stages_latency_ms["reranking"] = round((time.perf_counter() - t2) * 1000, 2)

        # 4. Context Compression Stage
        should_compress = self.use_compression if compress is None else compress
        t3 = time.perf_counter()
        if should_compress and candidates:
            final_results = self.compressor.compress(effective_query, candidates)
        else:
            final_results = candidates
        trace.stages_latency_ms["compression"] = round((time.perf_counter() - t3) * 1000, 2)

        trace.final_results_count = len(final_results)
        trace.total_latency_ms = round((time.perf_counter() - start_pipeline) * 1000, 2)

        return PipelineRetrievalResponse(results=final_results, trace=trace)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """BaseRetriever implementation returning final ranked SearchResult list."""
        response = self.retrieve_with_trace(query, top_k=top_k, filters=filters)
        return response.results
