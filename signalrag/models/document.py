"""Document schemas and metadata for SignalRAG."""

import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata associated with an ingested document."""

    source: str
    file_name: str
    file_type: str = "text"
    file_size_bytes: int | None = None
    content_hash: str | None = None
    title: str | None = None
    author: str | None = None
    total_pages: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Normalized document representation before chunking."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: DocumentMetadata

    @classmethod
    def from_text(
        cls,
        text: str,
        source: str,
        file_name: str | None = None,
        file_type: str = "text",
        title: str | None = None,
        **extra: Any,
    ) -> "Document":
        """Convenience factory method to instantiate a Document."""
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        name = file_name or source.split("/")[-1]
        metadata = DocumentMetadata(
            source=source,
            file_name=name,
            file_type=file_type,
            file_size_bytes=len(text.encode("utf-8")),
            content_hash=content_hash,
            title=title,
            extra=extra,
        )
        # Deterministic document ID based on hash and source
        doc_id = hashlib.sha256(f"{source}:{content_hash}".encode()).hexdigest()[:16]
        return cls(id=doc_id, text=text, metadata=metadata)
