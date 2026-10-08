"""Tests for document, chunk, retrieval, and citation models."""

from signalrag.models import (
    Chunk,
    ChunkMetadata,
    Citation,
    Document,
    SearchResult,
)


def test_document_creation():
    doc = Document.from_text(
        text="Sample document text for SignalRAG testing.",
        source="data/sample.txt",
        title="Sample Document",
        category="finance",
    )
    assert doc.id is not None
    assert doc.text == "Sample document text for SignalRAG testing."
    assert doc.metadata.source == "data/sample.txt"
    assert doc.metadata.file_name == "sample.txt"
    assert doc.metadata.content_hash is not None
    assert doc.metadata.title == "Sample Document"
    assert doc.metadata.extra["category"] == "finance"


def test_chunk_creation():
    chunk = Chunk.create(
        document_id="doc123",
        chunk_index=0,
        text="This is chunk 0 of the document.",
        source="data/sample.txt",
        page_number=2,
        start_char=0,
        end_char=32,
        section_title="Introduction",
    )
    assert chunk.document_id == "doc123"
    assert chunk.metadata.chunk_index == 0
    assert chunk.metadata.page_number == 2
    assert chunk.metadata.section_title == "Introduction"
    assert chunk.id is not None


def test_search_result_and_citation():
    meta = ChunkMetadata(
        document_id="doc1",
        chunk_index=1,
        source="reports/2024.pdf",
        page_number=5,
        section_title="Key Metrics",
    )
    chunk = Chunk(
        id="chunk1",
        document_id="doc1",
        text="Revenue grew by 24% year-over-year.",
        metadata=meta,
    )
    res = SearchResult(
        chunk=chunk,
        score=0.92,
        retrieval_method="hybrid",
        rank=1,
    )
    assert res.score == 0.92
    assert res.retrieval_method == "hybrid"

    citation = Citation(
        index=1,
        source=chunk.metadata.source,
        document_id=chunk.document_id,
        chunk_id=chunk.id,
        page_number=chunk.metadata.page_number,
        section_title=chunk.metadata.section_title,
        quote=chunk.text,
        relevance_score=res.score,
    )
    assert citation.render_marker() == "[1]"
    assert citation.render_source_line() == "[1] reports/2024.pdf, p. 5 (Key Metrics)"
