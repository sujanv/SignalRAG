"""Tests for the complete SignalRAGEngine product."""

from pathlib import Path

from typer.testing import CliRunner

from signalrag.cli.main import app
from signalrag.embeddings import DeterministicHashEmbeddingService
from signalrag.generation import MockLLMClient, SignalRAGEngine
from signalrag.indexing import MemoryVectorStore
from signalrag.models.chunk import Chunk
from signalrag.retrieval import BM25Retriever, RetrievalPipeline, SemanticRetriever

runner = CliRunner()


def test_signalrag_engine_ask_and_stream():
    store = MemoryVectorStore()
    emb_service = DeterministicHashEmbeddingService(dimension=64)

    c1 = Chunk.create(
        "d1",
        0,
        "SignalRAG is a production-grade Retrieval-Augmented Generation evaluation platform with hybrid search.",
        "arch.md",
        page_number=1,
    )
    chunks = [c1]
    emb_service.embed_chunks(chunks)
    store.add_chunks(chunks)

    semantic = SemanticRetriever(store, emb_service)
    bm25 = BM25Retriever(chunks)
    pipeline = RetrievalPipeline(semantic_retriever=semantic, bm25_retriever=bm25)

    engine = SignalRAGEngine(retrieval_pipeline=pipeline, llm_client=MockLLMClient())

    # 1. Synchronous ask
    response = engine.ask("What is SignalRAG?")
    assert response.answer is not None
    assert "SignalRAG" in response.answer
    assert "[1]" in response.answer
    assert len(response.citations) >= 1
    assert response.guardrail is not None
    assert "### Sources" in response.rendered_markdown

    # 2. Streaming ask
    stream_res = engine.stream_ask("What is SignalRAG?")
    streamed_text = stream_res.accumulate_text()
    assert "SignalRAG" in streamed_text
    assert "[1]" in streamed_text


def test_cli_query_command(tmp_path: Path):
    corpus_dir = tmp_path / "data"
    corpus_dir.mkdir()
    (corpus_dir / "rag_guide.md").write_text(
        "SignalRAG implements grounded answer generation with automatic citation verification.",
        encoding="utf-8",
    )
    storage_dir = tmp_path / "storage"

    # Index first
    runner.invoke(app, ["index", str(corpus_dir), "--storage-dir", str(storage_dir)])

    # Query with streaming
    res_query = runner.invoke(
        app,
        ["query", "What does SignalRAG implement?", "--storage-dir", str(storage_dir), "--stream"],
    )
    assert res_query.exit_code == 0
    assert "Question:" in res_query.stdout
    assert "Answer:" in res_query.stdout
    assert "Sources" in res_query.stdout
    assert "rag_guide.md" in res_query.stdout
