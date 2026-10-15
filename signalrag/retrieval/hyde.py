"""Hypothetical Document Embeddings (HyDE) mapping queries into passage embedding space."""

from __future__ import annotations

from pydantic import BaseModel

from signalrag.embeddings.base import BaseEmbeddingService
from signalrag.indexing.vector_store import BaseVectorStore
from signalrag.models.retrieval import SearchResult


class HypotheticalDocument(BaseModel):
    """Generated hypothetical answer passage representing document semantic structure."""

    query: str
    hypothetical_passage: str
    generation_mode: str = "zero_shot_heuristic"


class HyDERetriever:
    """Generates synthetic hypothetical answer passages to query vector stores in document-space."""

    TEMPLATES = [
        "In response to the question '{q}', the documentation states that the system is designed to provide comprehensive mechanisms for resolving the query accurately through standard protocols.",
        "A technical overview addressing '{q}': The architecture implements robust algorithms and structural components to ensure high performance and reliability.",
    ]

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_service: BaseEmbeddingService,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def generate_hypothetical_document(self, query: str) -> HypotheticalDocument:
        """Synthesize a representative hypothetical answer passage from the user question."""
        template = self.TEMPLATES[hash(query) % len(self.TEMPLATES)]
        passage = template.format(q=query.strip().rstrip("?"))
        return HypotheticalDocument(query=query, hypothetical_passage=passage)

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Synthesize hypothetical document, embed in document-space, and search vector store."""
        hypo = self.generate_hypothetical_document(query)
        # Embed hypothetical passage instead of original question
        hypo_vector = self.embedding_service.embed_text(hypo.hypothetical_passage)
        return self.vector_store.search(hypo_vector, top_k=top_k)
