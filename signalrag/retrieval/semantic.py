"""Dense semantic retriever utilizing vector similarity search."""

from typing import Any

from signalrag.embeddings.base import BaseEmbeddingService
from signalrag.indexing.vector_store import BaseVectorStore
from signalrag.models.retrieval import SearchResult
from signalrag.retrieval.base import BaseRetriever


class SemanticRetriever(BaseRetriever):
    """Dense retriever that maps queries to vector embeddings and performs nearest-neighbor search."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_service: BaseEmbeddingService,
        min_score: float | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Embed user query and query vector store for top-K candidates."""
        query_text = query.strip()
        if not query_text or top_k <= 0:
            return []

        query_vec = self.embedding_service.embed_text(query_text)
        results = self.vector_store.search(query_vec, top_k=top_k, filters=filters)

        # Filter by minimum score if configured
        if self.min_score is not None:
            results = [r for r in results if r.score >= self.min_score]

        # Ensure correct retrieval method tag and rank assignment
        for idx, res in enumerate(results, start=1):
            res.retrieval_method = "vector"
            res.rank = idx

        return results
