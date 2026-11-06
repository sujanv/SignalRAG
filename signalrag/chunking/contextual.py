"""Contextual retrieval enricher prepending situational document context headers to chunks."""

from __future__ import annotations

from signalrag.models.chunk import Chunk
from signalrag.models.document import Document


class ContextualChunkEnricher:
    """Enriches chunk text with concise document-level context to prevent information loss during chunking."""

    def __init__(self, max_context_chars: int = 150):
        self.max_context_chars = max_context_chars

    def extract_document_context(self, document: Document) -> str:
        """Extract a concise situational summary from document title and opening sentences."""
        if isinstance(document.metadata, dict):
            source = document.metadata.get("source", document.id)
        else:
            source = getattr(document.metadata, "source", document.id)
        opening = document.text[: self.max_context_chars].replace("\n", " ").strip()
        if len(document.text) > self.max_context_chars:
            opening += "..."
        return f"[Document: {source} | Subject: {opening}]"

    def enrich_chunk(self, chunk: Chunk, document: Document) -> Chunk:
        """Prepend situational context header to chunk and record metadata."""
        prefix = self.extract_document_context(document)
        enriched_text = f"{prefix}\n{chunk.text}"

        # Preserve original raw text in metadata
        chunk.metadata.extra["raw_text"] = chunk.text
        chunk.metadata.extra["contextual_prefix"] = prefix
        chunk.text = enriched_text
        return chunk

    def enrich_chunks(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        """Apply contextual enrichment across all chunks belonging to a document."""
        return [self.enrich_chunk(c, document) for c in chunks]
