"""Embedding service factory."""

from signalrag.core.config import EmbeddingConfig
from signalrag.embeddings.base import BaseEmbeddingService
from signalrag.embeddings.hash_embedding import DeterministicHashEmbeddingService
from signalrag.embeddings.local_embedding import SentenceTransformerEmbeddingService
from signalrag.embeddings.openai_embedding import OpenAIEmbeddingService


def create_embedding_service(config: EmbeddingConfig | None = None) -> BaseEmbeddingService:
    """Create embedding provider instance based on EmbeddingConfig."""
    cfg = config or EmbeddingConfig()

    provider = cfg.provider.lower()
    if provider in ("local", "hash", "mock"):
        return DeterministicHashEmbeddingService(dimension=cfg.dimension)
    elif provider in ("sentence_transformers", "fastembed"):
        return SentenceTransformerEmbeddingService(model_name=cfg.model, batch_size=cfg.batch_size)
    elif provider == "openai":
        return OpenAIEmbeddingService(
            model_name=cfg.model,
            api_key=cfg.api_key,
            dimension=cfg.dimension,
        )
    else:
        # Default to deterministic hash embedding for robustness
        return DeterministicHashEmbeddingService(dimension=cfg.dimension)
