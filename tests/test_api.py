"""Tests for SignalRAG REST API."""

from fastapi.testclient import TestClient

from signalrag.api.app import app

client = TestClient(app)


def test_health_check():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_create_and_get_document():
    create_res = client.post(
        "/documents",
        json={
            "id": "doc_api_1",
            "content": "RRF combines BM25 and vector search.",
            "metadata": {"tag": "rag"},
        },
    )
    assert create_res.status_code == 201
    assert create_res.json()["document_id"] == "doc_api_1"

    get_res = client.get("/documents/doc_api_1")
    assert get_res.status_code == 200
    assert get_res.json()["character_count"] > 10

    not_found = client.get("/documents/nonexistent_xyz")
    assert not_found.status_code == 404


def test_query_endpoint():
    res = client.post("/query", json={"query": "What is RRF?", "top_k": 3})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "latency_ms" in data


def test_index_endpoint():
    res = client.post("/index", json={"force_reindex": True})
    assert res.status_code == 200
    assert res.json()["status"] == "success"


def test_evaluations_endpoint():
    res = client.get("/evaluations")
    assert res.status_code == 200
    assert isinstance(res.json(), list)
