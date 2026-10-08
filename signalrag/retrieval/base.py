"""Base abstractions for search and retrieval."""

from abc import ABC, abstractmethod
from typing import Any

from signalrag.models.retrieval import SearchResult


class BaseRetriever(ABC):
    """Abstract interface for retrieving relevant candidate chunks from a query."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve top-K search results matching the query."""
        pass
