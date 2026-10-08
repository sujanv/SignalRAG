"""OpenAI API-compatible embedding provider."""

import os
from typing import Any

from signalrag.embeddings.base import BaseEmbeddingService


class OpenAIEmbeddingService(BaseEmbeddingService):
    """Embedding service using OpenAI API (text-embedding-3-small, etc.)."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        dimension: int = 1536,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self._dimension = dimension
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(
                    api_key=self.api_key or "mock-key",
                    base_url=self.base_url,
                )
            except ImportError as err:
                raise ImportError(
                    "openai package is not installed. Install via `pip install openai`."
                ) from err
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        results = self.embed_texts([text])
        return results[0] if results else [0.0] * self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        response = client.embeddings.create(input=texts, model=self.model_name)
        # Sort embeddings by return index
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
