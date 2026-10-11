"""Tests for Python client SDK."""

from fastapi.testclient import TestClient

from signalrag.api.app import app
from signalrag.client import SignalRAGClient


def test_python_client_flow():
    with TestClient(app) as http_client:
        client = SignalRAGClient(client=http_client)
        # 1. Add document
        doc_res = client.add_document(
            content="Enterprise retrieval using vector similarity and BM25 keywords.",
            metadata={"source": "client_test"},
        )
        assert doc_res.document_id is not None
        assert doc_res.character_count > 0

        # 2. Index
        idx_res = client.index()
        assert idx_res.status == "success"
        assert idx_res.total_chunks >= 1

        # 3. Query
        q_res = client.query("What does enterprise retrieval use?", top_k=2)
        assert q_res.answer is not None
        assert len(q_res.query) > 0

        # 4. Metrics
        metrics = client.get_metrics()
        assert isinstance(metrics, dict)
