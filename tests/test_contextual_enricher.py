"""Tests for Contextual Chunk Enrichment."""

from signalrag.chunking.contextual import ContextualChunkEnricher
from signalrag.chunking.recursive import RecursiveCharacterChunker
from signalrag.models.document import Document


def test_contextual_chunk_enrichment():
    doc = Document(
        id="arch_spec",
        text="SignalRAG distributed architecture guide covering BM25 and vector fusion across distributed shards.",
        metadata={"source": "arch_spec.pdf"},
    )

    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=10)
    raw_chunks = chunker.chunk_document(doc)
    assert len(raw_chunks) >= 1

    enricher = ContextualChunkEnricher()
    enriched = enricher.enrich_chunks(doc, raw_chunks)

    assert len(enriched) == len(raw_chunks)
    first_chunk = enriched[0]
    assert "[Document: arch_spec.pdf" in first_chunk.text
    assert "raw_text" in first_chunk.metadata.extra
    assert "contextual_prefix" in first_chunk.metadata.extra
