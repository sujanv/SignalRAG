"""Tests for Concurrent Batch Indexing."""

from pathlib import Path

from signalrag.chunking.recursive import RecursiveCharacterChunker
from signalrag.embeddings.hash_embedding import DeterministicHashEmbeddingService
from signalrag.indexing.batch import ConcurrentBatchIndexer
from signalrag.indexing.memory_store import MemoryVectorStore
from signalrag.models.document import Document


def test_concurrent_batch_indexing():
    docs = [
        Document(
            id="d1",
            text="Document one discussing high throughput retrieval systems and vector indexing architectures.",
            metadata={"source": "d1"},
        ),
        Document(
            id="d2",
            text="Document two detailing concurrent worker thread execution, batch processing, and low latency queries.",
            metadata={"source": "d2"},
        ),
    ]

    indexer = ConcurrentBatchIndexer(max_workers=2, batch_size=2)
    chunker = RecursiveCharacterChunker(chunk_size=60, chunk_overlap=10, min_chunk_size=10)
    emb = DeterministicHashEmbeddingService(dimension=16)
    store = MemoryVectorStore()

    report = indexer.index_documents(docs, chunker, emb, store)

    assert report.documents_processed == 2
    assert report.chunks_created >= 2
    assert report.elapsed_seconds >= 0.0
    assert store.count() == report.chunks_created


def test_go_batch_cli_source():
    go_file = Path("clients/go/cmd/signalrag-batch/main.go")
    assert go_file.exists()
    assert "package main" in go_file.read_text()
