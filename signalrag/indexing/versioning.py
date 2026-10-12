"""Document version tracking, historical diffs, and temporal retrieval filtering."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from signalrag.models.chunk import Chunk, ChunkMetadata


class DocumentRevision(BaseModel):
    """Snapshot of a document version."""

    document_id: str
    version: int
    created_at: float = Field(default_factory=time.time)
    chunk_ids: list[str] = Field(default_factory=list)
    summary: str = ""
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentVersionManager:
    """Manages document versions and filters chunks based on temporal constraints."""

    def __init__(self):
        # Mapping: document_id -> list of revisions in ascending order
        self._revisions: dict[str, list[DocumentRevision]] = {}
        # Mapping: chunk_id -> (document_id, version)
        self._chunk_version_map: dict[str, tuple[str, int]] = {}

    def register_version(
        self,
        document_id: str,
        chunks: list[Chunk],
        summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> DocumentRevision:
        """Register a new document version, automatically superseding previous versions."""
        revs = self._revisions.setdefault(document_id, [])
        new_version_num = len(revs) + 1

        # Mark prior revisions as inactive
        for r in revs:
            r.is_active = False

        chunk_ids = [c.id for c in chunks]
        rev = DocumentRevision(
            document_id=document_id,
            version=new_version_num,
            created_at=time.time(),
            chunk_ids=chunk_ids,
            summary=summary,
            is_active=True,
            metadata=metadata or {},
        )
        revs.append(rev)

        for cid in chunk_ids:
            self._chunk_version_map[cid] = (document_id, new_version_num)

        # Update metadata on chunks
        for c in chunks:
            if not isinstance(c.metadata, ChunkMetadata):
                continue
            c.metadata.extra["doc_version"] = new_version_num
            c.metadata.extra["is_active_version"] = True

        return rev

    def get_latest_version(self, document_id: str) -> DocumentRevision | None:
        revs = self._revisions.get(document_id, [])
        return revs[-1] if revs else None

    def get_version_history(self, document_id: str) -> list[DocumentRevision]:
        return list(self._revisions.get(document_id, []))

    def filter_active_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Filter out chunks belonging to superseded document versions."""
        active_chunks = []
        for c in chunks:
            info = self._chunk_version_map.get(c.id)
            if not info:
                # If unversioned, keep by default
                active_chunks.append(c)
                continue
            doc_id, ver = info
            latest = self.get_latest_version(doc_id)
            if latest and latest.version == ver and latest.is_active:
                active_chunks.append(c)
        return active_chunks

    def filter_as_of(self, chunks: list[Chunk], timestamp: float) -> list[Chunk]:
        """Return chunks that were valid at a specific point in time."""
        result = []
        for c in chunks:
            info = self._chunk_version_map.get(c.id)
            if not info:
                result.append(c)
                continue
            doc_id, ver = info
            revs = self._revisions.get(doc_id, [])
            valid_rev = None
            for r in revs:
                if r.created_at <= timestamp:
                    valid_rev = r
                else:
                    break
            if valid_rev and valid_rev.version == ver:
                result.append(c)
        return result
