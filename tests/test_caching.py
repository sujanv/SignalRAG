"""Tests for LRUCache and CachedEmbeddingService."""

import time

import numpy as np
import pytest

from signalrag.embeddings.cached_embedding import CachedEmbeddingService
from signalrag.embeddings.hash_embedding import DeterministicHashEmbeddingService
from signalrag.retrieval.cache import LRUCache, QueryCache


def test_lru_cache_operations():
    cache = LRUCache(max_size=2, ttl_seconds=10.0)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.hits == 2

    # Eviction
    cache.set("c", 3)
    assert cache.get("a") is None  # 'a' was oldest
    assert cache.misses == 1
    assert cache.hit_rate == pytest.approx(2 / 3, rel=1e-2)


def test_lru_cache_ttl():
    cache = LRUCache(max_size=10, ttl_seconds=0.05)
    cache.set("short_lived", "data")
    assert cache.get("short_lived") == "data"
    time.sleep(0.06)
    assert cache.get("short_lived") is None


def test_query_cache():
    qc = QueryCache()
    assert qc.get_query_result("What is RAG?", top_k=5) is None
    qc.set_query_result("What is RAG?", top_k=5, result="Response payload")
    assert qc.get_query_result("What is RAG?", top_k=5) == "Response payload"
    assert qc.get_query_result("What is RAG?", top_k=10) is None


def test_cached_embedding_service():
    base = DeterministicHashEmbeddingService(dimension=16)
    cached = CachedEmbeddingService(base, cache_size=100)

    vec1 = cached.embed_text("Sample input query")
    assert cached.cache.misses == 1
    assert cached.cache.hits == 0

    vec2 = cached.embed_text("Sample input query")
    assert cached.cache.hits == 1
    assert np.allclose(vec1, vec2)

    batch_vecs = cached.embed_batch(["Sample input query", "New query"])
    assert len(batch_vecs) == 2
    assert cached.cache.hits == 2
