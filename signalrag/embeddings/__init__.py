"""Embedding service interfaces and providers."""

from signalrag.embeddings.base import BaseEmbeddingService, cosine_similarity
from signalrag.embeddings.factory import create_embedding_service
from signalrag.embeddings.hash_embedding import DeterministicHashEmbeddingService
from signalrag.embeddings.local_embedding import SentenceTransformerEmbeddingService
from signalrag.embeddings.openai_embedding import OpenAIEmbeddingService

__all__ = [
    "BaseEmbeddingService",
    "cosine_similarity",
    "DeterministicHashEmbeddingService",
    "SentenceTransformerEmbeddingService",
    "OpenAIEmbeddingService",
    "create_embedding_service",
]
