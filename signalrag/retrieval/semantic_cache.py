"""Semantic query caching layer evaluating vector cosine proximity to prior queries."""

from __future__ import annotations

import time

import numpy as np
from pydantic import BaseModel, Field

from signalrag.models.retrieval import SearchResult


class CacheEntry(BaseModel):
    """Cached query results with embedding and temporal expiration."""

    query: str
    query_vector: list[float]
    results: list[SearchResult]
    created_at: float = Field(default_factory=time.time)
    ttl_seconds: float = 3600.0

    def is_expired(self, now: float | None = None) -> bool:
        current = now or time.time()
        return (current - self.created_at) > self.ttl_seconds


class SemanticQueryCache:
    """Caches query results and matches semantically proximate queries using cosine similarity."""

    def __init__(self, similarity_threshold: float = 0.90, default_ttl: float = 3600.0):
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
        self._entries: list[CacheEntry] = []
        self.hits = 0
        self.misses = 0

    def get(
        self,
        query_vector: list[float],
        similarity_threshold: float | None = None,
    ) -> list[SearchResult] | None:
        """Lookup cached results if any prior query exceeds the cosine similarity threshold."""
        threshold = similarity_threshold or self.similarity_threshold
        now = time.time()

        # Clean expired
        self._entries = [e for e in self._entries if not e.is_expired(now)]

        if not self._entries or not query_vector:
            self.misses += 1
            return None

        q_arr = np.array(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_arr))
        if q_norm == 0:
            self.misses += 1
            return None

        best_score = -1.0
        best_entry: CacheEntry | None = None

        for entry in self._entries:
            e_arr = np.array(entry.query_vector, dtype=np.float32)
            e_norm = float(np.linalg.norm(e_arr))
            if e_norm == 0:
                continue
            sim = float(np.dot(q_arr, e_arr) / (q_norm * e_norm))
            if sim > best_score:
                best_score = sim
                best_entry = entry

        if best_entry and best_score >= threshold:
            self.hits += 1
            return best_entry.results

        self.misses += 1
        return None

    def set(
        self,
        query: str,
        query_vector: list[float],
        results: list[SearchResult],
        ttl_seconds: float | None = None,
    ) -> None:
        """Store query and results in semantic cache."""
        entry = CacheEntry(
            query=query,
            query_vector=query_vector,
            results=results,
            created_at=time.time(),
            ttl_seconds=ttl_seconds or self.default_ttl,
        )
        self._entries.append(entry)

    def stats(self) -> dict[str, float | int]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "cached_entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 3),
        }

    def clear(self) -> None:
        self._entries.clear()
