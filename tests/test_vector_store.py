"""Tests for vector store indexing, similarity search, filtering, and persistence."""

from pathlib import Path

from signalrag.embeddings import DeterministicHashEmbeddingService
from signalrag.indexing import MemoryVectorStore
from signalrag.models.chunk import Chunk


def test_memory_vector_store_add_and_search():
    store = MemoryVectorStore()
    emb_service = DeterministicHashEmbeddingService(dimension=64)

    c1 = Chunk.create("doc1", 0, "Deep neural networks for natural language processing", "doc1.txt")
    c2 = Chunk.create("doc2", 0, "Convolutional neural networks for computer vision", "doc2.txt")
    c3 = Chunk.create("doc3", 0, "Culinary art of baking sourdough bread and pastries", "doc3.txt")

    emb_service.embed_chunks([c1, c2, c3])
    store.add_chunks([c1, c2, c3])

    assert store.count() == 3

    # Search query closely related to NLP
    query_vec = emb_service.embed_text("natural language processing transformers")
    results = store.search(query_vec, top_k=2)

    assert len(results) == 2
    assert results[0].chunk.id == c1.id
    assert results[0].rank == 1
    assert results[0].score > results[1].score


def test_vector_store_metadata_filtering():
    store = MemoryVectorStore()
    emb_service = DeterministicHashEmbeddingService(dimension=64)

    c1 = Chunk.create("doc1", 0, "Revenue metrics for Q1", "finance.pdf", author="Finance Team")
    c2 = Chunk.create("doc2", 0, "Revenue metrics for Q2", "finance.pdf", author="Finance Team")
    c3 = Chunk.create("doc3", 0, "Engineering sprint metrics", "eng.pdf", author="Eng Team")

    emb_service.embed_chunks([c1, c2, c3])
    store.add_chunks([c1, c2, c3])

    query_vec = emb_service.embed_text("revenue metrics")

    # Filter by author
    eng_results = store.search(query_vec, top_k=5, filters={"author": "Eng Team"})
    assert len(eng_results) == 1
    assert eng_results[0].chunk.id == c3.id

    # Filter by source
    fin_results = store.search(query_vec, top_k=5, filters={"source": "finance.pdf"})
    assert len(fin_results) == 2


def test_vector_store_save_and_load(tmp_path: Path):
    save_file = tmp_path / "index.json"
    store = MemoryVectorStore(storage_path=save_file)
    emb_service = DeterministicHashEmbeddingService(dimension=32)

    c1 = Chunk.create("doc1", 0, "Vector store persistence test", "test.txt")
    emb_service.embed_chunks([c1])
    store.add_chunks([c1])

    store.save()
    assert save_file.exists()

    # Load in new store instance
    loaded_store = MemoryVectorStore(storage_path=save_file)
    loaded_store.load()

    assert loaded_store.count() == 1
    loaded_chunk = loaded_store.get_chunk(c1.id)
    assert loaded_chunk is not None
    assert loaded_chunk.text == c1.text


def test_vector_store_delete():
    store = MemoryVectorStore()
    emb_service = DeterministicHashEmbeddingService(dimension=32)

    c1 = Chunk.create("doc1", 0, "Text 1", "test.txt")
    c2 = Chunk.create("doc2", 0, "Text 2", "test.txt")
    emb_service.embed_chunks([c1, c2])
    store.add_chunks([c1, c2])

    assert store.count() == 2
    deleted = store.delete([c1.id])
    assert deleted == 1
    assert store.count() == 1
    assert store.get_chunk(c1.id) is None
    assert store.get_chunk(c2.id) is not None
