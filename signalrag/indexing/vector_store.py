"""Base interface for vector stores."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult


class BaseVectorStore(ABC):
    """Abstract interface for storing and searching chunk vector embeddings."""

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks to the index. Returns count of chunks added."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Perform nearest-neighbor vector similarity search with optional metadata filters."""
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve chunk by ID."""
        pass

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks from index by ID. Returns number deleted."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of chunks indexed."""
        pass

    @abstractmethod
    def save(self, target_path: str | Path | None = None) -> None:
        """Persist vector index to storage."""
        pass

    @abstractmethod
    def load(self, source_path: str | Path | None = None) -> None:
        """Load vector index from storage."""
        pass
