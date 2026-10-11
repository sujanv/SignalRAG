"""Official Python client SDK for SignalRAG API."""

from __future__ import annotations

from typing import Any

import httpx

from signalrag.api.schemas import (
    DocumentResponse,
    IndexResponse,
    QueryResponse,
)


class SignalRAGClient:
    """Synchronous HTTP client for interacting with SignalRAG service."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        if client is not None:
            self._client = client
        else:
            self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def query(self, query: str, top_k: int = 5) -> QueryResponse:
        """Execute a RAG query and return answer with citations."""
        res = self._client.post("/query", json={"query": query, "top_k": top_k})
        res.raise_for_status()
        return QueryResponse.model_validate(res.json())

    def add_document(
        self, content: str, document_id: str | None = None, metadata: dict[str, Any] | None = None
    ) -> DocumentResponse:
        """Ingest a new raw text document."""
        payload: dict[str, Any] = {"content": content, "metadata": metadata or {}}
        if document_id:
            payload["document_id"] = document_id
        res = self._client.post("/documents", json=payload)
        res.raise_for_status()
        return DocumentResponse.model_validate(res.json())

    def index(self) -> IndexResponse:
        """Trigger vector and lexical indexing of ingested documents."""
        res = self._client.post("/index", json={})
        res.raise_for_status()
        return IndexResponse.model_validate(res.json())

    def get_metrics(self) -> dict[str, Any]:
        """Fetch server operational latency and token metrics."""
        res = self._client.get("/metrics")
        res.raise_for_status()
        return res.json()

    def close(self) -> None:
        try:
            self._client.close()
        except AttributeError:
            pass

    def __enter__(self) -> SignalRAGClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
