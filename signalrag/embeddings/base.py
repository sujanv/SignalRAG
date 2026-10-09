"""Base embedding service abstraction."""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

from signalrag.models.chunk import Chunk


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimensions do not match: {len(v1)} != {len(v2)}")

    dot = sum(a * b for a, b in zip(v1, v2, strict=False))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(dot / (norm_a * norm_b))


class BaseEmbeddingService(ABC):
    """Abstract base class for text embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single string into a vector."""
        pass

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings. Subclasses can override for vectorization."""
        return [self.embed_text(t) for t in texts]

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed a list of Chunks in place and set chunk.embedding."""
        if not chunks:
            return chunks

        texts = [c.text for c in chunks]
        embeddings = self.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings, strict=False):
            chunk.embedding = emb

        return chunks


# Backward compatibility alias
EmbeddingService = BaseEmbeddingService
