"""Tests for Semantic Query Caching."""

from signalrag.models.chunk import Chunk, ChunkMetadata
from signalrag.models.retrieval import SearchResult
from signalrag.retrieval.semantic_cache import SemanticQueryCache


def test_semantic_cache_hit_and_miss():
    cache = SemanticQueryCache(similarity_threshold=0.85)

    c = Chunk(
        id="c1",
        document_id="d1",
        text="Cached result text",
        metadata=ChunkMetadata(document_id="d1", chunk_index=0, source="s1"),
    )
    res = [SearchResult(chunk=c, score=1.0)]

    # Populate cache
    cache.set("What is SignalRAG?", [1.0, 0.0, 0.0], res)

    # Near semantic query vector (similarity ~0.99)
    cached = cache.get([0.99, 0.02, 0.0])
    assert cached is not None
    assert cached[0].chunk.id == "c1"
    assert cache.hits == 1

    # Orthogonal query vector (similarity 0.0) -> Miss
    miss = cache.get([0.0, 1.0, 0.0])
    assert miss is None
    assert cache.misses == 1


def test_semantic_cache_ttl():
    cache = SemanticQueryCache(similarity_threshold=0.8)
    c = Chunk(
        id="c1",
        document_id="d1",
        text="text",
        metadata=ChunkMetadata(document_id="d1", chunk_index=0, source="s1"),
    )
    cache.set("query", [1.0, 0.0], [SearchResult(chunk=c, score=1.0)], ttl_seconds=0.01)

    import time

    time.sleep(0.02)
    assert cache.get([1.0, 0.0]) is None
