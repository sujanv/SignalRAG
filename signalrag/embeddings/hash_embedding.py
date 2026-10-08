"""Deterministic token-hash embedding model for offline testing and fast local execution."""

import hashlib
import math
import re

from signalrag.embeddings.base import BaseEmbeddingService


class DeterministicHashEmbeddingService(BaseEmbeddingService):
    """Deterministic, lightweight embedding model using n-gram token hashing and unit normalization.

    Generates dense embeddings where documents with lexical and semantic overlap
    have positive cosine similarity, without requiring external model weights or internet access.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_token(self, token: str, seed: int = 0) -> int:
        """Hash a token string to an integer index within embedding dimension."""
        key = f"{token}:{seed}".encode()
        h = int(hashlib.md5(key).hexdigest()[:8], 16)
        return h % self._dimension

    def embed_text(self, text: str) -> list[float]:
        """Convert text into a normalized dense vector."""
        if not text:
            return [0.0] * self._dimension

        vector = [0.0] * self._dimension
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return [0.0] * self._dimension

        # 1. Word unigrams
        for token in tokens:
            idx1 = self._hash_token(token, seed=1)
            idx2 = self._hash_token(token, seed=2)
            vector[idx1] += 1.0
            vector[idx2] += 0.5

        # 2. Word bigrams
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i + 1]}"
            idx = self._hash_token(bigram, seed=3)
            vector[idx] += 1.5

        # 3. Character 3-grams
        for i in range(len(text) - 2):
            char_trigram = text[i : i + 3].lower()
            idx = self._hash_token(char_trigram, seed=4)
            vector[idx] += 0.2

        # L2 Normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0.0:
            vector = [v / norm for v in vector]

        return vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]
