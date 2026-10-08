"""Search and retrieval result schemas."""

from typing import Literal

from pydantic import BaseModel, Field

from signalrag.models.chunk import Chunk


class SearchResult(BaseModel):
    """Ranked search result containing the retrieved chunk and match signals."""

    chunk: Chunk
    score: float
    retrieval_method: Literal["vector", "bm25", "hybrid", "reranked"] = "vector"
    rank: int | None = None
    rerank_score: float | None = None
    metadata_filters_applied: dict[str, str] = Field(default_factory=dict)
