"""Pydantic request and response models for SignalRAG HTTP REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from signalrag.models.citation import CitationSource


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query to process through RAG pipeline")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of context chunks to retrieve")
    stream: bool = Field(default=False, description="Whether to stream response tokens")


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationSource] = Field(default_factory=list)
    latency_ms: float
    retrieved_chunk_count: int


class DocumentCreateRequest(BaseModel):
    id: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    document_id: str
    character_count: int
    metadata: Any = None


class IndexRequest(BaseModel):
    force_reindex: bool = False


class IndexResponse(BaseModel):
    indexed_documents: int
    total_chunks: int
    status: str


class EvaluationRunRequest(BaseModel):
    dataset_path: str = "eval/questions.json"
    limit: int | None = None
    top_k: int = 5
