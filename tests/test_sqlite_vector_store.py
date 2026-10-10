"""Tests for SQLite-backed persistent vector store."""

import tempfile
from pathlib import Path

from signalrag.indexing.sqlite_store import SQLiteVectorStore
from signalrag.models.chunk import Chunk, ChunkMetadata


def test_sqlite_vector_store_crud():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "vectors.db"
        store = SQLiteVectorStore(db_path=db_path)

        c1 = Chunk(
            id="c1",
            document_id="doc1",
            text="First chunk about Python programming",
            metadata=ChunkMetadata(
                document_id="doc1", chunk_index=0, source="doc1.txt", extra={"tag": "python"}
            ),
            embedding=[1.0, 0.0, 0.0],
        )
        c2 = Chunk(
            id="c2",
            document_id="doc2",
            text="Second chunk about Golang concurrent systems",
            metadata=ChunkMetadata(
                document_id="doc2", chunk_index=1, source="doc2.txt", extra={"tag": "go"}
            ),
            embedding=[0.0, 1.0, 0.0],
        )

        added = store.add_chunks([c1, c2])
        assert added == 2
        assert store.count() == 2
        assert len(store) == 2
        assert bool(store) is True

        retrieved = store.get_chunk("c1")
        assert retrieved is not None
        assert retrieved.text == c1.text
        assert retrieved.embedding == [1.0, 0.0, 0.0]

        # Persistence check
        store.save()
        store.close()

        store2 = SQLiteVectorStore(db_path=db_path)
        assert store2.count() == 2
        results = store2.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].chunk.id == "c1"
        assert round(results[0].score, 3) == 1.0

        # Delete check
        deleted = store2.delete(["c1"])
        assert deleted == 1
        assert store2.count() == 1
        assert store2.get_chunk("c1") is None
        store2.close()
