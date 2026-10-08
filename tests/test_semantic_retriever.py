"""Tests for dense semantic vector retriever."""

from signalrag.embeddings import DeterministicHashEmbeddingService
from signalrag.indexing import MemoryVectorStore
from signalrag.models.chunk import Chunk
from signalrag.retrieval import SemanticRetriever


def test_semantic_retriever_ranking():
    store = MemoryVectorStore()
    embedding_service = DeterministicHashEmbeddingService(dimension=64)

    c1 = Chunk.create("d1", 0, "Transformer models and attention mechanisms in deep learning", "ml.txt")
    c2 = Chunk.create("d2", 0, "Computer vision convolution algorithms and image filters", "cv.txt")
    c3 = Chunk.create("d3", 0, "French cuisine recipe with butter and shallots", "food.txt")

    embedding_service.embed_chunks([c1, c2, c3])
    store.add_chunks([c1, c2, c3])

    retriever = SemanticRetriever(vector_store=store, embedding_service=embedding_service)

    results = retriever.retrieve("transformers and multi-head attention", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.id == c1.id
    assert results[0].rank == 1
    assert results[0].retrieval_method == "vector"
    assert results[0].score >= results[1].score


def test_semantic_retriever_min_score():
    store = MemoryVectorStore()
    embedding_service = DeterministicHashEmbeddingService(dimension=64)

    c1 = Chunk.create("d1", 0, "Artificial intelligence research", "ai.txt")
    embedding_service.embed_chunks([c1])
    store.add_chunks([c1])

    # With high threshold, low matches are dropped
    retriever_strict = SemanticRetriever(
        vector_store=store,
        embedding_service=embedding_service,
        min_score=0.9999,
    )
    res = retriever_strict.retrieve("completely unrelated recipe", top_k=5)
    assert len(res) == 0


def test_semantic_retriever_empty_query():
    store = MemoryVectorStore()
    embedding_service = DeterministicHashEmbeddingService(dimension=64)
    retriever = SemanticRetriever(vector_store=store, embedding_service=embedding_service)

    assert retriever.retrieve("", top_k=5) == []
    assert retriever.retrieve("   ", top_k=5) == []
