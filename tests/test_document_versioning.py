"""Tests for document versioning and temporal filtering."""

from signalrag.indexing.versioning import DocumentVersionManager
from signalrag.models.chunk import Chunk, ChunkMetadata


def test_document_version_lifecycle():
    mgr = DocumentVersionManager()

    c1 = Chunk(
        id="c1_v1",
        document_id="doc1",
        text="Version 1 text about company policy.",
        metadata=ChunkMetadata(document_id="doc1", chunk_index=0, source="policy.txt"),
    )
    rev1 = mgr.register_version("doc1", [c1], summary="Initial revision")
    assert rev1.version == 1
    assert rev1.is_active is True

    # Register version 2
    c2 = Chunk(
        id="c1_v2",
        document_id="doc1",
        text="Version 2 updated policy terms.",
        metadata=ChunkMetadata(document_id="doc1", chunk_index=0, source="policy.txt"),
    )
    rev2 = mgr.register_version("doc1", [c2], summary="Updated terms")
    assert rev2.version == 2
    assert rev2.is_active is True
    assert rev1.is_active is False

    # Filter active chunks
    filtered = mgr.filter_active_chunks([c1, c2])
    assert len(filtered) == 1
    assert filtered[0].id == "c1_v2"


def test_version_history_and_temporal():
    mgr = DocumentVersionManager()
    c1 = Chunk(
        id="chunk_a",
        document_id="doc_a",
        text="Historic data",
        metadata=ChunkMetadata(document_id="doc_a", chunk_index=0, source="doc.txt"),
    )
    rev = mgr.register_version("doc_a", [c1])
    history = mgr.get_version_history("doc_a")
    assert len(history) == 1
    assert history[0].version == 1

    # Temporal query
    t_after = rev.created_at + 10.0
    valid_chunks = mgr.filter_as_of([c1], t_after)
    assert len(valid_chunks) == 1
