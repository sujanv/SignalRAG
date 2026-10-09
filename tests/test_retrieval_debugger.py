"""Tests for RetrievalDebugger."""

from rich.console import Console

from signalrag.core.config import SignalRAGConfig
from signalrag.generation.engine import RAGEngine
from signalrag.models.document import Document
from signalrag.retrieval.debugger import RetrievalDebugger, render_debug_trace


def test_retrieval_debugger():
    cfg = SignalRAGConfig()
    cfg.embeddings.provider = "hash"
    cfg.embeddings.dimensions = 32
    engine = RAGEngine.from_config(cfg)

    # Index doc
    doc = Document(
        document_id="doc_dbg",
        content="BM25 and vector search are merged using Reciprocal Rank Fusion.",
        metadata={"title": "RAG Deep Dive"},
    )
    engine.indexing_pipeline.index_documents([doc])

    debugger = RetrievalDebugger(engine.retrieval_pipeline)
    trace = debugger.trace("What is Reciprocal Rank Fusion?", top_k=3)

    assert trace.original_query == "What is Reciprocal Rank Fusion?"
    assert trace.total_latency_ms >= 0.0
    assert len(trace.merged_candidates) >= 1

    console = Console(record=True)
    render_debug_trace(trace, console=console)
    output = console.export_text()
    assert "SignalRAG Retrieval Debugger" in output
    assert "Rewritten Query" in output
