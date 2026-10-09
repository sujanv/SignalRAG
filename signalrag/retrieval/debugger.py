"""Retrieval debugger for inspecting intermediate representations across pipeline stages."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from signalrag.models.retrieval import SearchResult
from signalrag.retrieval.pipeline import RetrievalPipeline


class CandidateItem(BaseModel):
    """Snapshot of candidate chunk at a pipeline stage."""

    chunk_id: str
    score: float
    content_snippet: str
    metadata: Any = None


class DebugTrace(BaseModel):
    """Complete multi-stage retrieval execution trace."""

    original_query: str
    rewritten_query: str
    bm25_candidates: list[CandidateItem] = Field(default_factory=list)
    vector_candidates: list[CandidateItem] = Field(default_factory=list)
    merged_candidates: list[CandidateItem] = Field(default_factory=list)
    reranked_candidates: list[CandidateItem] = Field(default_factory=list)
    compressed_contexts: list[str] = Field(default_factory=list)
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0


class RetrievalDebugger:
    """Executes a retrieval pipeline with full diagnostic interception."""

    def __init__(self, pipeline: RetrievalPipeline):
        self.pipeline = pipeline

    def trace(self, query: str, top_k: int = 5) -> DebugTrace:
        """Execute query while recording candidate snapshots at every pipeline boundary."""
        t_total_start = time.perf_counter()
        latencies: dict[str, float] = {}

        # 1. Query Rewrite
        t0 = time.perf_counter()
        raw_rewritten = query
        if self.pipeline.query_rewriter:
            res_rw = self.pipeline.query_rewriter.rewrite(query)
            raw_rewritten = res_rw[0] if isinstance(res_rw, list) and res_rw else str(res_rw)
        effective_query = raw_rewritten
        latencies["rewrite"] = (time.perf_counter() - t0) * 1000

        # 2. BM25 Retrieval
        t0 = time.perf_counter()
        bm25_results = []
        if self.pipeline.bm25_retriever:
            bm25_results = self.pipeline.bm25_retriever.retrieve(effective_query, top_k=top_k * 2)
        latencies["bm25"] = (time.perf_counter() - t0) * 1000

        # 3. Vector Retrieval
        t0 = time.perf_counter()
        vector_results = []
        if self.pipeline.semantic_retriever:
            vector_results = self.pipeline.semantic_retriever.retrieve(
                effective_query, top_k=top_k * 2
            )
        latencies["vector"] = (time.perf_counter() - t0) * 1000

        # 4. Hybrid Merge
        t0 = time.perf_counter()
        merged_results = self.pipeline.hybrid_retriever.retrieve(effective_query, top_k=top_k * 2)
        latencies["hybrid_merge"] = (time.perf_counter() - t0) * 1000

        # 5. Reranking
        t0 = time.perf_counter()
        reranked_results = merged_results
        if self.pipeline.reranker:
            reranked_results = self.pipeline.reranker.rerank(
                query=effective_query, results=merged_results, top_k=top_k
            )
        latencies["rerank"] = (time.perf_counter() - t0) * 1000

        # 6. Context Compression
        t0 = time.perf_counter()
        compressor = getattr(self.pipeline, "compressor", None) or getattr(
            self.pipeline, "context_compressor", None
        )
        if compressor:
            compressed_res = compressor.compress(query=effective_query, results=reranked_results)
            compressed = [r.chunk.content for r in compressed_res]
        else:
            compressed = [r.chunk.content for r in reranked_results]
        latencies["compression"] = (time.perf_counter() - t0) * 1000

        total_lat = (time.perf_counter() - t_total_start) * 1000

        def to_items(results: list[SearchResult]) -> list[CandidateItem]:
            return [
                CandidateItem(
                    chunk_id=r.chunk.chunk_id,
                    score=round(r.score, 4),
                    content_snippet=r.chunk.content[:80].replace("\n", " "),
                    metadata=r.chunk.metadata,
                )
                for r in results
            ]

        return DebugTrace(
            original_query=query,
            rewritten_query=effective_query,
            bm25_candidates=to_items(bm25_results),
            vector_candidates=to_items(vector_results),
            merged_candidates=to_items(merged_results),
            reranked_candidates=to_items(reranked_results),
            compressed_contexts=compressed,
            stage_latencies_ms={k: round(v, 2) for k, v in latencies.items()},
            total_latency_ms=round(total_lat, 2),
        )


def render_debug_trace(trace: DebugTrace, console: Console | None = None) -> None:
    """Render the ASCII/Rich pipeline execution flow."""
    c = console or Console()

    tree = Tree(f"[bold cyan]QUERY:[/bold cyan] {trace.original_query}")
    tree.add(
        f"[dim]↓[/dim] [bold magenta]Rewritten Query:[/bold magenta] {trace.rewritten_query} "
        f"[dim]({trace.stage_latencies_ms.get('rewrite', 0)}ms)[/dim]"
    )

    bm25_node = tree.add(
        f"[dim]↓[/dim] [bold green]BM25 Results[/bold green] [dim]({trace.stage_latencies_ms.get('bm25', 0)}ms)[/dim]"
    )
    for item in trace.bm25_candidates[:3]:
        bm25_node.add(
            f"[cyan]{item.chunk_id}[/cyan] (score: {item.score}) -> {item.content_snippet}..."
        )

    vec_node = tree.add(
        f"[dim]↓[/dim] [bold green]Vector Results[/bold green] [dim]({trace.stage_latencies_ms.get('vector', 0)}ms)[/dim]"
    )
    for item in trace.vector_candidates[:3]:
        vec_node.add(
            f"[cyan]{item.chunk_id}[/cyan] (score: {item.score}) -> {item.content_snippet}..."
        )

    merge_node = tree.add(
        f"[dim]↓[/dim] [bold yellow]Merged Results (RRF/Linear)[/bold yellow] [dim]({trace.stage_latencies_ms.get('hybrid_merge', 0)}ms)[/dim]"
    )
    for item in trace.merged_candidates[:3]:
        merge_node.add(f"[cyan]{item.chunk_id}[/cyan] (fused: {item.score})")

    rerank_node = tree.add(
        f"[dim]↓[/dim] [bold red]Reranked Results[/bold red] [dim]({trace.stage_latencies_ms.get('rerank', 0)}ms)[/dim]"
    )
    for item in trace.reranked_candidates[:3]:
        rerank_node.add(f"[bold]{item.chunk_id}[/bold] (rerank score: {item.score})")

    ctx_node = tree.add(
        f"[dim]↓[/dim] [bold blue]Final Context[/bold blue] [dim]({trace.stage_latencies_ms.get('compression', 0)}ms)[/dim]"
    )
    for i, ctx in enumerate(trace.compressed_contexts[:3], start=1):
        ctx_node.add(f"Passage [{i}]: {ctx[:100]}...")

    c.print(
        Panel(
            tree,
            title="[bold green]SignalRAG Retrieval Debugger[/bold green]",
            subtitle=f"[dim]Total pipeline latency: {trace.total_latency_ms:.1f}ms[/dim]",
            box=ROUNDED,
        )
    )
