"""Tests for document ingestion base loaders."""

from pathlib import Path

import pytest

from signalrag.ingestion import DirectoryLoader, TextLoader


def test_text_loader_basic(tmp_path: Path):
    doc_path = tmp_path / "sample.md"
    doc_path.write_text("# Overview\nThis is a sample markdown document.", encoding="utf-8")

    loader = TextLoader(doc_path)
    docs = loader.load()

    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata.title == "Overview"
    assert doc.metadata.file_type == "md"
    assert doc.metadata.content_hash is not None
    assert "Overview" in doc.text


def test_text_loader_not_found():
    with pytest.raises(FileNotFoundError):
        TextLoader("non_existent_file.txt")


def test_directory_loader(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "file1.txt").write_text("Hello world", encoding="utf-8")
    (tmp_path / "sub" / "file2.md").write_text("Nested document", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"\x00\x01\x02")

    loader = DirectoryLoader(tmp_path)
    docs = loader.load()

    assert len(docs) == 2
    sources = [d.metadata.file_name for d in docs]
    assert "file1.txt" in sources
    assert "file2.md" in sources
