"""Tests for chunk metadata extraction, section tracking, and page mapping."""

from signalrag.chunking import ChunkMetadataExtractor, RecursiveCharacterChunker
from signalrag.models.document import Document


def test_chunk_metadata_section_and_hierarchy():
    text = (
        "# System Architecture\n\n"
        "SignalRAG is built as a modular retrieval and generation system.\n\n"
        "## Vector Indexing\n\n"
        "The vector indexing pipeline converts parsed documents into chunks and embeddings. "
        * 5
        + "\n\n"
        + "## Hybrid Search\n\n"
        "Hybrid search merges sparse lexical scores with dense embedding cosine similarities. " * 5
    )
    doc = Document.from_text(
        text=text,
        source="docs/arch.md",
        file_name="arch.md",
        file_type="md",
        author="Sujan V",
    )

    chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=30, min_chunk_size=30)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 3

    # Check section title and hierarchy
    section_titles = [c.metadata.section_title for c in chunks if c.metadata.section_title]
    assert any("System Architecture" in s or "Vector Indexing" in s for s in section_titles)

    # Check author inheritance
    for c in chunks:
        assert c.metadata.extra.get("author") == "Sujan V"
        assert c.metadata.start_char >= 0
        assert c.metadata.end_char <= len(text)
        assert c.metadata.token_count is not None and c.metadata.token_count > 0


def test_chunk_metadata_page_boundary_tracking():
    text = (
        "--- Page 1 ---\n"
        "First page content with financial executive summary and overview metrics.\n\n"
        "--- Page 2 ---\n"
        "Second page content with detailed breakdowns and revenue figures."
    )
    doc = Document.from_text(
        text=text,
        source="reports/annual.pdf",
        file_name="annual.pdf",
        file_type="pdf",
    )

    extractor = ChunkMetadataExtractor(track_pages=True)
    chunker = RecursiveCharacterChunker(
        chunk_size=100, chunk_overlap=20, min_chunk_size=20, metadata_extractor=extractor
    )
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    pages = [c.metadata.page_number for c in chunks]
    assert 1 in pages
    assert 2 in pages
