"""Embedding service proxy caching vector representations by content hash."""

from __future__ import annotations

import hashlib

from signalrag.embeddings.base import BaseEmbeddingService
from signalrag.retrieval.cache import LRUCache


class CachedEmbeddingService(BaseEmbeddingService):
    """Wraps an EmbeddingService with an in-memory SHA256 embedding cache."""

    def __init__(self, delegate: BaseEmbeddingService, cache_size: int = 5000):
        self.delegate = delegate
        self.cache = LRUCache(max_size=cache_size, ttl_seconds=86400.0)

    @property
    def dimension(self) -> int:
        return self.delegate.dimension

    @property
    def dimensions(self) -> int:
        return self.delegate.dimension

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_text(self, text: str) -> list[float]:
        key = self._hash(text)
        cached = self.cache.get(key)
        if cached is not None:
            return list(cached)

        vector = self.delegate.embed_text(text)
        self.cache.set(key, vector)
        return vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = []
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for idx, t in enumerate(texts):
            key = self._hash(t)
            cached = self.cache.get(key)
            if cached is not None:
                results.append(list(cached))
            else:
                results.append(None)
                missing_indices.append(idx)
                missing_texts.append(t)

        if missing_texts:
            fresh_vectors = self.delegate.embed_texts(missing_texts)
            for orig_idx, vec, t in zip(
                missing_indices, fresh_vectors, missing_texts, strict=False
            ):
                key = self._hash(t)
                self.cache.set(key, vec)
                results[orig_idx] = vec

        return [r for r in results if r is not None]

    embed_batch = embed_texts
