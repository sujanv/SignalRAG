"""Tests for PDF parsing pipeline."""

from pathlib import Path

import pytest
from pypdf import PageObject, PdfWriter

from signalrag.ingestion.pdf_loader import PDFLoader, normalize_pdf_text


def test_normalize_pdf_text():
    raw = "This is some hy-\nphenated text.\r\nLine 2 has   multiple   spaces.\n\n\n\nParagraph 2."
    cleaned = normalize_pdf_text(raw)
    assert "hyphenated text." in cleaned
    assert "Line 2 has multiple spaces." in cleaned
    assert "\n\nParagraph 2." in cleaned


def create_sample_pdf(file_path: Path, pages: list[str], title: str = "Test Doc", author: str = "Test Author"):
    """Helper to generate a valid PDF for testing without network."""
    writer = PdfWriter()
    for _text in pages:
        # Create a blank page
        page = PageObject.create_blank_page(width=300, height=300)
        writer.add_page(page)

    writer.add_metadata({
        "/Title": title,
        "/Author": author,
    })
    with file_path.open("wb") as f:
        writer.write(f)


def test_pdf_loader_document_mode(tmp_path: Path):
    pdf_file = tmp_path / "sample.pdf"
    create_sample_pdf(pdf_file, ["Page 1 content", "Page 2 content"], title="Research Paper")

    loader = PDFLoader(pdf_file, mode="document")
    docs = loader.load()

    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata.title == "Research Paper"
    assert doc.metadata.total_pages == 2
    assert doc.metadata.file_type == "pdf"
    assert doc.metadata.file_size_bytes is not None
    assert doc.metadata.source == str(pdf_file)


def test_pdf_loader_page_mode(tmp_path: Path):
    pdf_file = tmp_path / "multipage.pdf"
    create_sample_pdf(pdf_file, ["P1", "P2", "P3"], title="Multipage Document")

    loader = PDFLoader(pdf_file, mode="page")
    docs = loader.load()

    assert len(docs) == 3
    for idx, doc in enumerate(docs, start=1):
        assert doc.metadata.extra["page_number"] == idx
        assert doc.metadata.total_pages == 3


def test_pdf_loader_corrupt_file(tmp_path: Path):
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"%PDF-1.4 garbage data that is not a valid pdf")

    loader = PDFLoader(corrupt_file)
    with pytest.raises(ValueError, match="Malformed or corrupted PDF file"):
        loader.load()
