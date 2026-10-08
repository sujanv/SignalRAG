"""Chunk schema and metadata."""

import hashlib
import uuid
from typing import Any

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata describing a specific document chunk."""

    document_id: str
    chunk_index: int
    source: str
    page_number: int | None = None
    start_char: int = 0
    end_char: int = 0
    section_title: str | None = None
    token_count: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Atomic text chunk with embedding and positional metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    text: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None

    @classmethod
    def create(
        cls,
        document_id: str,
        chunk_index: int,
        text: str,
        source: str,
        page_number: int | None = None,
        start_char: int = 0,
        end_char: int = 0,
        section_title: str | None = None,
        token_count: int | None = None,
        embedding: list[float] | None = None,
        **extra: Any,
    ) -> "Chunk":
        """Factory helper creating a Chunk with deterministic ID."""
        chunk_hash = hashlib.sha256(f"{document_id}:{chunk_index}:{text}".encode()).hexdigest()[:16]
        meta = ChunkMetadata(
            document_id=document_id,
            chunk_index=chunk_index,
            source=source,
            page_number=page_number,
            start_char=start_char,
            end_char=end_char,
            section_title=section_title,
            token_count=token_count,
            extra=extra,
        )
        return cls(
            id=chunk_hash,
            document_id=document_id,
            text=text,
            metadata=meta,
            embedding=embedding,
        )
