"""SentenceTransformers local embedding provider."""

from typing import Any

from signalrag.embeddings.base import BaseEmbeddingService


class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """Local transformer embedding service backed by sentence-transformers or fastembed."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = None
        self._dimension: int | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError as err:
                raise ImportError(
                    "sentence-transformers is not installed. Run `pip install sentence-transformers` "
                    "or configure EMBEDDING_PROVIDER=mock / local."
                ) from err
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension or 384

    def embed_text(self, text: str) -> list[float]:
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [e.tolist() for e in embeddings]
