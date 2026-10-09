"""Central configuration management for SignalRAG."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    """Configuration for LLM generation."""

    provider: Literal["openai", "anthropic", "ollama", "gemini", "mock"] = "openai"
    model: str = "gpt-4o-mini"
    api_key: str | None = None

    @property
    def dimensions(self) -> int:
        return self.dimension

    @dimensions.setter
    def dimensions(self, val: int) -> None:
        self.dimension = val

    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    timeout_seconds: float = Field(default=30.0, gt=0)


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    provider: Literal["local", "fastembed", "openai", "mock", "hash"] = "local"
    model: str = "all-MiniLM-L6-v2"
    dimension: int = Field(default=384, gt=0)
    batch_size: int = Field(default=32, gt=0)
    api_key: str | None = None

    @property
    def dimensions(self) -> int:
        return self.dimension

    @dimensions.setter
    def dimensions(self, val: int) -> None:
        self.dimension = val


class VectorStoreConfig(BaseModel):
    """Configuration for vector database / index persistence."""

    type: Literal["memory", "lancedb", "chroma"] = "lancedb"
    path: Path = Path("./storage/vector_store")
    collection_name: str = "signalrag_chunks"


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""

    strategy: Literal["recursive", "token", "sentence"] = "recursive"
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    min_chunk_size: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        return self


class RetrievalConfig(BaseModel):
    """Configuration for query retrieval and reranking."""

    top_k: int = Field(default=5, gt=0)
    hybrid_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for vector search in hybrid mode: 1.0 = pure vector, 0.0 = pure BM25",
    )
    use_bm25: bool = True
    use_semantic: bool = True
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = Field(default=3, gt=0)
    compression_enabled: bool = False


class EvaluationConfig(BaseModel):
    """Configuration for RAG evaluation benchmarks."""

    dataset_path: Path | None = None
    output_dir: Path = Path("./storage/eval_results")
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])


class Settings(BaseSettings):
    """Root configuration for SignalRAG."""

    model_config = SettingsConfigDict(
        env_prefix="SIGNALRAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "SignalRAG"
    debug: bool = False
    log_level: str = "INFO"

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @property
    def embeddings(self) -> EmbeddingConfig:
        return self.embedding

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Settings":
        """Load settings from a YAML configuration file."""
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        return cls(**data)

    def to_yaml(self, target_path: str | Path | None = None) -> str:
        """Serialize settings to YAML."""
        data = self.model_dump(mode="json")
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
        if target_path:
            path = Path(target_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml_str, encoding="utf-8")
        return yaml_str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


SignalRAGConfig = Settings
load_config = Settings.from_yaml
