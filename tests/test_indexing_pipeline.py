"""Tests for the end-to-end indexing pipeline."""

from pathlib import Path

from signalrag.chunking import RecursiveCharacterChunker
from signalrag.embeddings import DeterministicHashEmbeddingService
from signalrag.indexing import IndexingPipeline, MemoryVectorStore
from signalrag.models.document import Document


def test_indexing_pipeline_documents():
    chunker = RecursiveCharacterChunker(chunk_size=150, chunk_overlap=30, min_chunk_size=20)
    embedding_service = DeterministicHashEmbeddingService(dimension=64)
    vector_store = MemoryVectorStore()

    pipeline = IndexingPipeline(
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    doc1 = Document.from_text(
        text="# Neural Attention Models\nTransformers use multi-head self-attention mechanisms to encode sequence relations. " * 3,
        source="doc1.md",
        title="Attention Models",
    )
    doc2 = Document.from_text(
        text="# Italian Cooking\nTraditional Neapolitan pizza dough is fermented slowly with double-zero flour. " * 3,
        source="doc2.md",
        title="Pizza Guide",
    )

    res = pipeline.index_documents([doc1, doc2])
    assert res.total_documents == 2
    assert res.total_chunks > 2
    assert res.duration_seconds >= 0.0
    assert vector_store.count() == res.total_chunks

    # Search for attention
    results = pipeline.search("self-attention transformers", top_k=2)
    assert len(results) > 0
    assert results[0].chunk.document_id == doc1.id
    assert "Attention" in (results[0].chunk.metadata.section_title or "")


def test_indexing_pipeline_index_path(tmp_path: Path):
    doc_file = tmp_path / "rag.md"
    doc_file.write_text(
        "# SignalRAG Architecture\nHybrid retrieval balances semantic vectors with lexical BM25 signals.",
        encoding="utf-8",
    )

    pipeline = IndexingPipeline()
    res = pipeline.index_path(tmp_path)

    assert res.total_documents == 1
    assert res.total_chunks >= 1

    search_res = pipeline.search("hybrid retrieval BM25", top_k=1)
    assert len(search_res) == 1
    assert "SignalRAG" in search_res[0].chunk.text
