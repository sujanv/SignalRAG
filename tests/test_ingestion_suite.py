"""Comprehensive ingestion test suite covering fixtures, malformed docs, metadata, and CLI."""

from pathlib import Path

import pytest
from pypdf import PageObject, PdfWriter
from typer.testing import CliRunner

from signalrag.cli.main import app
from signalrag.ingestion import DirectoryLoader

runner = CliRunner()


@pytest.fixture
def complex_corpus_dir(tmp_path: Path) -> Path:
    """Fixture creating a realistic directory structure with valid, malformed, and edge-case documents."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    # 1. Valid markdown document
    (corpus / "guide.md").write_text(
        "# SignalRAG Architecture Guide\nSignalRAG implements production-grade RAG evaluation.",
        encoding="utf-8",
    )

    # 2. Valid nested text file
    sub_dir = corpus / "research"
    sub_dir.mkdir()
    (sub_dir / "notes.txt").write_text(
        "Notes on hybrid retrieval and BM25 indexing.", encoding="utf-8"
    )

    # 3. Valid PDF file
    pdf_path = sub_dir / "paper.pdf"
    writer = PdfWriter()
    p1 = PageObject.create_blank_page(width=200, height=200)
    p2 = PageObject.create_blank_page(width=200, height=200)
    writer.add_page(p1)
    writer.add_page(p2)
    writer.add_metadata({"/Title": "RAG Evaluation Benchmarks", "/Author": "SignalRAG Team"})
    with pdf_path.open("wb") as f:
        writer.write(f)

    # 4. Hidden file (should be ignored)
    (corpus / ".secret.txt").write_text("Hidden config secret", encoding="utf-8")

    # 5. Unsupported file type (should be ignored by default loaders)
    (corpus / "binary.bin").write_bytes(b"\x00\xff\xee\xdd")

    return corpus


def test_directory_ingestion_corpus(complex_corpus_dir: Path):
    loader = DirectoryLoader(complex_corpus_dir)
    docs = loader.load()

    assert len(docs) == 3

    file_names = {d.metadata.file_name for d in docs}
    assert file_names == {"guide.md", "notes.txt", "paper.pdf"}

    # Ensure hidden files and binary files are skipped
    assert ".secret.txt" not in file_names
    assert "binary.bin" not in file_names

    # Check metadata integrity
    pdf_doc = next(d for d in docs if d.metadata.file_name == "paper.pdf")
    assert pdf_doc.metadata.title == "RAG Evaluation Benchmarks"
    assert pdf_doc.metadata.author == "SignalRAG Team"
    assert pdf_doc.metadata.total_pages == 2
    assert pdf_doc.metadata.content_hash is not None


def test_malformed_document_error_handling(tmp_path: Path):
    bad_dir = tmp_path / "bad_corpus"
    bad_dir.mkdir()

    # Good file
    (bad_dir / "good.txt").write_text("Good text", encoding="utf-8")

    # Corrupted PDF
    (bad_dir / "corrupted.pdf").write_bytes(b"NOT A REAL PDF %PDF-invalid")

    # By default, silent_errors=False raises RuntimeError
    loader_strict = DirectoryLoader(bad_dir, silent_errors=False)
    with pytest.raises(RuntimeError, match="Failed to load file"):
        loader_strict.load()

    # With silent_errors=True, skip corrupt file and load good file
    loader_silent = DirectoryLoader(bad_dir, silent_errors=True)
    loaded_docs = loader_silent.load()
    assert len(loaded_docs) == 1
    assert loaded_docs[0].metadata.file_name == "good.txt"


def test_cli_ingest_command(complex_corpus_dir: Path):
    result = runner.invoke(app, ["ingest", str(complex_corpus_dir)])
    assert result.exit_code == 0
    assert "Ingested Documents (3 total)" in result.stdout
    assert "Successfully ingested 3 structured documents." in result.stdout


def test_cli_ingest_single_pdf(complex_corpus_dir: Path):
    pdf_path = complex_corpus_dir / "research" / "paper.pdf"
    result = runner.invoke(app, ["ingest", str(pdf_path), "--page-mode"])
    assert result.exit_code == 0
    assert "Ingested Documents (2 total)" in result.stdout
    assert "Successfully ingested 2 structured documents." in result.stdout


def test_cli_ingest_nonexistent_path():
    result = runner.invoke(app, ["ingest", "/nonexistent/path/signalrag"])
    assert result.exit_code == 1
    assert "Path '/nonexistent/path/signalrag' does not exist." in result.stdout
