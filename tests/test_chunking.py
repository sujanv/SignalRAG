"""Tests for document chunking."""

import pytest

from signalrag.chunking import RecursiveCharacterChunker, TokenChunker
from signalrag.models.document import Document


def test_recursive_chunker_basic():
    text = (
        "Introduction to SignalRAG.\n\n"
        "SignalRAG is designed for production RAG evaluation. " * 10
        + "\n\n"
        + "Chapter 2: Retrieval techniques. " * 10
    )
    chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=40, min_chunk_size=20)
    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 260  # Allows small boundary tolerance


def test_recursive_chunk_document():
    doc = Document.from_text(
        text="Section 1.\n\n" + "This is a detailed paragraph about vector search. " * 8,
        source="docs/vector.txt",
        title="Vector Search Guide",
    )
    chunker = RecursiveCharacterChunker(chunk_size=150, chunk_overlap=30, min_chunk_size=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.document_id == doc.id
        assert chunk.metadata.source == "docs/vector.txt"
        assert chunk.metadata.token_count is not None
        assert chunk.metadata.token_count > 0
        assert chunk.metadata.start_char >= 0
        assert chunk.metadata.end_char > chunk.metadata.start_char


def test_token_chunker():
    text = "SignalRAG uses token windowing for exact LLM context fitting. " * 20
    chunker = TokenChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=10)
    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1
    tokens = chunker.tokenizer.encode(chunks[0])
    assert len(tokens) <= 50


def test_chunker_overlap_validation():
    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=120)
