"""Tests for incremental indexing and CLI search."""

from pathlib import Path

from typer.testing import CliRunner

from signalrag.cli.main import app
from signalrag.indexing import IncrementalIndexer, IndexingPipeline, MemoryVectorStore
from signalrag.models.document import Document

runner = CliRunner()


def test_incremental_indexing_skip_unchanged(tmp_path: Path):
    index_file = tmp_path / "index.json"
    ledger_file = tmp_path / "ledger.json"

    vector_store = MemoryVectorStore(storage_path=index_file)
    pipeline = IndexingPipeline(vector_store=vector_store)
    indexer = IncrementalIndexer(pipeline=pipeline, ledger_path=ledger_file)

    doc = Document.from_text(
        text="Dense retrieval uses continuous vector spaces to match semantic queries. " * 3,
        source="dense.txt",
    )

    # First pass: adds new document
    res1 = indexer.index([doc])
    assert res1.added_documents == 1
    assert res1.skipped_documents == 0
    assert res1.added_chunks > 0

    # Second pass: unchanged document should be skipped
    res2 = indexer.index([doc])
    assert res2.added_documents == 0
    assert res2.skipped_documents == 1
    assert res2.added_chunks == 0


def test_incremental_indexing_updates_modified_doc(tmp_path: Path):
    index_file = tmp_path / "index.json"
    ledger_file = tmp_path / "ledger.json"

    vector_store = MemoryVectorStore(storage_path=index_file)
    pipeline = IndexingPipeline(vector_store=vector_store)
    indexer = IncrementalIndexer(pipeline=pipeline, ledger_path=ledger_file)

    doc_v1 = Document.from_text(
        text="Version 1 of the architecture design document.",
        source="design.txt",
    )
    indexer.index([doc_v1])
    initial_chunk_count = vector_store.count()

    doc_v2 = Document.from_text(
        text="Version 2 of the architecture design document with expanded multi-head attention details.",
        source="design.txt",
    )
    res_update = indexer.index([doc_v2])

    assert res_update.updated_documents == 1
    assert res_update.added_documents == 0
    assert res_update.skipped_documents == 0
    assert vector_store.count() >= initial_chunk_count


def test_cli_index_and_search(tmp_path: Path):
    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    (data_dir / "quantum.txt").write_text(
        "Quantum computing harnesses superposition and entanglement to solve specific computational problems exponentially faster.",
        encoding="utf-8",
    )

    storage_dir = tmp_path / "storage"

    # Run CLI index
    res_idx = runner.invoke(app, ["index", str(data_dir), "--storage-dir", str(storage_dir)])
    assert res_idx.exit_code == 0
    assert "Incremental Indexing" in res_idx.stdout
    assert "Newly Added Docs" in res_idx.stdout

    # Re-run index to verify incremental skip
    res_idx2 = runner.invoke(app, ["index", str(data_dir), "--storage-dir", str(storage_dir)])
    assert res_idx2.exit_code == 0
    assert "Skipped (Unchanged)" in res_idx2.stdout

    # Run CLI search
    res_search = runner.invoke(
        app,
        ["search", "quantum superposition", "--storage-dir", str(storage_dir)],
    )
    assert res_search.exit_code == 0
    assert "Semantic Search Results" in res_search.stdout
    assert "quantum.txt" in res_search.stdout
