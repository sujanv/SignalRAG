"""Persistent SQLite-backed vector store with cosine similarity and metadata indexing."""

from __future__ import annotations

import json
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from signalrag.indexing.vector_store import BaseVectorStore
from signalrag.models.chunk import Chunk, ChunkMetadata
from signalrag.models.retrieval import SearchResult


def _floats_to_blob(vector: list[float]) -> bytes:
    """Pack float array into binary blob."""
    return struct.pack(f"{len(vector)}f", *vector)


def _blob_to_floats(blob: bytes) -> list[float]:
    """Unpack float array from binary blob."""
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


class SQLiteVectorStore(BaseVectorStore):
    """ACID-compliant persistent vector store utilizing SQLite for metadata and vector storage."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id)")

    def count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks")
        res = cur.fetchone()
        return res[0] if res else 0

    def __len__(self) -> int:
        return self.count()

    def __bool__(self) -> bool:
        return True

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, document_id, text, metadata_json, embedding_blob FROM chunks WHERE id = ?",
            (chunk_id,),
        )
        row = cur.fetchone()
        if not row:
            return None

        meta_dict = json.loads(row["metadata_json"])
        meta = ChunkMetadata(**meta_dict)
        embedding = _blob_to_floats(row["embedding_blob"])

        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            text=row["text"],
            metadata=meta,
            embedding=embedding,
        )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        valid_chunks = [c for c in chunks if c.embedding is not None]
        if not valid_chunks:
            raise ValueError("All chunks added to vector store must have non-null embeddings.")

        now = time.time()
        rows = []
        for c in valid_chunks:
            assert c.embedding is not None
            meta_json = json.dumps(c.metadata.model_dump(mode="json"))
            emb_blob = _floats_to_blob(c.embedding)
            rows.append((c.id, c.document_id, c.text, meta_json, emb_blob, len(c.embedding), now))

        with self.conn:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                (id, document_id, text, metadata_json, embedding_blob, dim, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def delete(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        cur = self.conn.cursor()
        placeholders = ",".join("?" for _ in chunk_ids)
        with self.conn:
            cur.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", chunk_ids)
        return cur.rowcount

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, document_id, text, metadata_json, embedding_blob FROM chunks")
        rows = cur.fetchall()

        if not rows:
            return []

        q_arr = np.array(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_arr))
        if q_norm == 0:
            q_norm = 1e-9

        candidates: list[tuple[float, Chunk]] = []
        for r in rows:
            meta_dict = json.loads(r["metadata_json"])
            if filters and not self._matches_filter(meta_dict, filters):
                continue

            emb = _blob_to_floats(r["embedding_blob"])
            emb_arr = np.array(emb, dtype=np.float32)
            emb_norm = float(np.linalg.norm(emb_arr))
            if emb_norm == 0:
                emb_norm = 1e-9

            cosine_sim = float(np.dot(q_arr, emb_arr) / (q_norm * emb_norm))
            chunk = Chunk(
                id=r["id"],
                document_id=r["document_id"],
                text=r["text"],
                metadata=ChunkMetadata(**meta_dict),
                embedding=emb,
            )
            candidates.append((cosine_sim, chunk))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [SearchResult(chunk=c, score=score) for score, c in candidates[:top_k]]

    def _matches_filter(self, meta: dict[str, Any], filters: dict[str, Any]) -> bool:
        for k, v in filters.items():
            if k not in meta or meta[k] != v:
                return False
        return True

    def save(self, target_path: str | Path | None = None) -> None:
        self.conn.commit()

    def load(self, source_path: str | Path | None = None) -> None:
        pass

    def close(self) -> None:
        self.conn.close()
