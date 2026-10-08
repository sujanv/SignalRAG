"""Tests for context compression and the end-to-end retrieval pipeline."""

from pathlib import Path

from typer.testing import CliRunner

from signalrag.cli.main import app
from signalrag.embeddings import DeterministicHashEmbeddingService
from signalrag.indexing import MemoryVectorStore
from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult
from signalrag.retrieval import (
    BM25Retriever,
    RetrievalPipeline,
    SemanticRetriever,
    SentenceRelevanceCompressor,
)

runner = CliRunner()


def test_sentence_relevance_compressor():
    long_text = (
        "This is an introductory preamble sentence that is not relevant. "
        "Dense vector retrieval uses continuous embeddings for semantic match. "
        "Another sentence discussing unrelated breakfast foods and coffee. "
        "High dimensional embeddings capture contextual representations well."
    )
    c1 = Chunk.create("d1", 0, long_text, "test.txt")
    res = SearchResult(chunk=c1, score=0.8, rank=1)

    compressor = SentenceRelevanceCompressor(min_sentence_score=0.1, max_sentences_per_chunk=2)
    compressed = compressor.compress("dense vector retrieval embeddings", [res])

    assert len(compressed) == 1
    compressed_text = compressed[0].chunk.text
    assert len(compressed_text) < len(long_text)
    assert "Dense vector retrieval" in compressed_text
    assert "breakfast foods" not in compressed_text
    assert "compression_ratio" in compressed[0].chunk.metadata.extra
    assert compressed[0].chunk.metadata.extra["compression_ratio"] < 1.0


def test_retrieval_pipeline_end_to_end():
    store = MemoryVectorStore()
    emb_service = DeterministicHashEmbeddingService(dimension=64)

    c1 = Chunk.create(
        "d1",
        0,
        "Introduction to SignalRAG. SignalRAG provides complete RAG evaluation metrics like MRR and NDCG. Additional filler text.",
        "signalrag.md",
        section_title="Evaluation Metrics",
    )
    c2 = Chunk.create(
        "d2",
        0,
        "Unrelated document regarding ancient Roman architecture and aqueducts.",
        "history.md",
    )

    chunks = [c1, c2]
    emb_service.embed_chunks(chunks)
    store.add_chunks(chunks)

    semantic = SemanticRetriever(store, emb_service)
    bm25 = BM25Retriever(chunks)

    pipeline = RetrievalPipeline(
        semantic_retriever=semantic,
        bm25_retriever=bm25,
    )

    # Query with conversational filler to test rewriter -> hybrid -> reranker -> compressor
    response = pipeline.retrieve_with_trace(
        query="Can you explain what are the SignalRAG evaluation metrics?",
        top_k=1,
        rerank=True,
        compress=True,
    )

    assert len(response.results) == 1
    top_result = response.results[0]
    assert top_result.chunk.id == c1.id
    assert top_result.rank == 1

    # Verify diagnostic trace
    trace = response.trace
    assert trace.raw_query.startswith("Can you explain")
    assert len(trace.rewritten_queries) >= 1
    assert "hybrid_retrieval" in trace.stages_latency_ms
    assert "reranking" in trace.stages_latency_ms
    assert "compression" in trace.stages_latency_ms
    assert trace.total_latency_ms > 0.0


def test_cli_search_with_hybrid_rerank_compress_debug(tmp_path: Path):
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    (data_dir / "doc.txt").write_text(
        "SignalRAG evaluates retrieval recall at K and mean reciprocal rank MRR. More background context here.",
        encoding="utf-8",
    )
    storage_dir = tmp_path / "storage"

    # Index first
    runner.invoke(app, ["index", str(data_dir), "--storage-dir", str(storage_dir)])

    # Search with all pipeline flags
    res = runner.invoke(
        app,
        [
            "search",
            "What is MRR in SignalRAG?",
            "--storage-dir",
            str(storage_dir),
            "--rerank",
            "--compress",
            "--debug",
        ],
    )

    assert res.exit_code == 0
    assert "Retrieval Pipeline Diagnostic Trace" in res.stdout
    assert "Retrieval Results for" in res.stdout
    assert "doc.txt" in res.stdout
