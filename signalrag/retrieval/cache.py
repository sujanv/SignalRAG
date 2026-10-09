"""Caching infrastructure for query responses and retrieval candidates."""

from __future__ import annotations

import collections
import hashlib
import time
from typing import Any


class LRUCache:
    """Thread-safe in-memory LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: collections.OrderedDict[str, tuple[Any, float]] = collections.OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        """Retrieve key if present and not expired."""
        if key not in self._cache:
            self.misses += 1
            return None

        val, exp = self._cache[key]
        if time.time() > exp:
            del self._cache[key]
            self.misses += 1
            return None

        self._cache.move_to_end(key)
        self.hits += 1
        return val

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Insert or update key with TTL expiration."""
        expiration = time.time() + (ttl if ttl is not None else self.ttl_seconds)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, expiration)

        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear cache contents."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total > 0 else 0.0


class QueryCache:
    """Specialized cache for repeated search queries and RAG responses."""

    def __init__(self, max_size: int = 500, ttl_seconds: float = 1800.0):
        self.cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)

    def _hash_query(self, query: str, top_k: int) -> str:
        raw = f"{query.strip().lower()}::topk={top_k}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_query_result(self, query: str, top_k: int) -> Any | None:
        key = self._hash_query(query, top_k)
        return self.cache.get(key)

    def set_query_result(self, query: str, top_k: int, result: Any) -> None:
        key = self._hash_query(query, top_k)
        self.cache.set(key, result)
