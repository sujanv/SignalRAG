"""In-memory numpy vector store with metadata filtering and persistence."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from signalrag.indexing.vector_store import BaseVectorStore
from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult


class MemoryVectorStore(BaseVectorStore):
    """Numpy-accelerated in-memory vector store with persistence to JSON/NumPy."""

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self._chunks: dict[str, Chunk] = {}
        self._ids: list[str] = []
        self._embeddings: np.ndarray | None = None  # shape: (N, D)

    def count(self) -> int:
        return len(self._chunks)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks and update vector matrix."""
        if not chunks:
            return 0

        valid_chunks = [c for c in chunks if c.embedding is not None]
        if not valid_chunks:
            raise ValueError("All chunks added to vector store must have non-null embeddings.")

        new_embeddings = [c.embedding for c in valid_chunks]
        new_mat = np.array(new_embeddings, dtype=np.float32)

        # Normalize to unit vectors for cosine similarity via dot product
        norms = np.linalg.norm(new_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        normalized_mat = new_mat / norms

        if self._embeddings is None or len(self._ids) == 0:
            self._embeddings = normalized_mat
        else:
            self._embeddings = np.vstack([self._embeddings, normalized_mat])

        for c in valid_chunks:
            self._chunks[c.id] = c
            self._ids.append(c.id)

        return len(valid_chunks)

    def _matches_filters(self, chunk: Chunk, filters: dict[str, Any] | None) -> bool:
        """Evaluate equality, comparison, and operator filters using MetadataFilter."""
        if not filters:
            return True
        from signalrag.retrieval.filter import MetadataFilter

        return MetadataFilter(filters).matches(chunk)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Perform cosine similarity search using normalized matrix multiplication."""
        if self._embeddings is None or len(self._ids) == 0 or top_k <= 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec = q_vec / q_norm

        # Cosine similarity dot product: (N, D) @ (D,) -> (N,)
        scores = np.dot(self._embeddings, q_vec)

        # Get sorted candidate indices
        sorted_indices = np.argsort(-scores)

        results: list[SearchResult] = []
        rank = 1
        for idx in sorted_indices:
            chunk_id = self._ids[idx]
            chunk = self._chunks[chunk_id]

            if not self._matches_filters(chunk, filters):
                continue

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(scores[idx]),
                    retrieval_method="vector",
                    rank=rank,
                    metadata_filters_applied={k: str(v) for k, v in (filters or {}).items()},
                )
            )
            rank += 1
            if len(results) >= top_k:
                break

        return results

    def delete(self, chunk_ids: list[str]) -> int:
        """Remove chunks by IDs and rebuild index."""
        ids_to_remove = set(chunk_ids)
        if not ids_to_remove:
            return 0

        keep_indices = [i for i, cid in enumerate(self._ids) if cid not in ids_to_remove]
        deleted_count = len(self._ids) - len(keep_indices)

        for cid in ids_to_remove:
            self._chunks.pop(cid, None)

        self._ids = [self._ids[i] for i in keep_indices]
        if self._embeddings is not None and len(keep_indices) > 0:
            self._embeddings = self._embeddings[keep_indices]
        else:
            self._embeddings = None

        return deleted_count

    def save(self, target_path: str | Path | None = None) -> None:
        """Persist chunk data and embeddings to file."""
        dest = Path(target_path or self.storage_path or "./storage/vector_store/index.json")
        dest.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "ids": self._ids,
            "chunks": [chunk.model_dump(mode="json") for chunk in self._chunks.values()],
        }

        # Write metadata & chunks json
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Write numpy matrix alongside
        if self._embeddings is not None:
            npy_path = dest.with_suffix(".npy")
            np.save(npy_path, self._embeddings)

    def load(self, source_path: str | Path | None = None) -> None:
        """Load vector index from persistence file."""
        src = Path(source_path or self.storage_path or "./storage/vector_store/index.json")
        if not src.exists():
            raise FileNotFoundError(f"Index storage file not found: {src}")

        payload = json.loads(src.read_text(encoding="utf-8"))
        self._chunks.clear()
        self._ids = payload.get("ids", [])

        for c_dict in payload.get("chunks", []):
            chunk = Chunk(**c_dict)
            self._chunks[chunk.id] = chunk

        npy_path = src.with_suffix(".npy")
        if npy_path.exists():
            self._embeddings = np.load(npy_path)
        else:
            # Reconstruct embeddings array from chunks
            embs = [self._chunks[cid].embedding for cid in self._ids if self._chunks[cid].embedding]
            if embs:
                self._embeddings = np.array(embs, dtype=np.float32)
            else:
                self._embeddings = None
