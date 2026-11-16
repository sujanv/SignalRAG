"""Document-aware semantic cache invalidation ledger and selective key purging."""

from __future__ import annotations

import collections
import time
from pydantic import BaseModel, Field

from signalrag.retrieval.semantic_cache import SemanticQueryCache


class InvalidationReport(BaseModel):
    """Telemetry report after selective cache invalidation."""

    document_id: str
    purged_keys: list[str] = Field(default_factory=list)
    remaining_entries: int = 0
    timestamp: float = Field(default_factory=time.time)


class CacheInvalidationLedger:
    """Tracks which cache entries depend on specific document IDs and handles selective cache invalidation."""

    def __init__(self):
        # document_id -> set of cache query strings
        self.doc_to_queries: dict[str, set[str]] = collections.defaultdict(set)

    def register_dependency(self, document_id: str, query: str) -> None:
        """Register that a cached query response depends on a specific source document."""
        self.doc_to_queries[document_id].add(query)

    def invalidate_document(
        self,
        document_id: str,
        cache: SemanticQueryCache,
    ) -> InvalidationReport:
        """Purge all cached query entries that depended on the modified document."""
        dependent_queries = self.doc_to_queries.get(document_id, set())
        purged = []

        remaining = []
        for entry in cache._entries:
            dep_match = entry.query in dependent_queries
            chunk_match = any(
                document_id in (r.chunk.document_id or r.chunk.id)
                for r in entry.results
                if hasattr(r, "chunk") and r.chunk
            )
            if dep_match or chunk_match:
                purged.append(entry.query)
            else:
                remaining.append(entry)

        cache._entries = remaining
        self.doc_to_queries.pop(document_id, None)

        return InvalidationReport(
            document_id=document_id,
            purged_keys=purged,
            remaining_entries=len(cache._entries),
        )
