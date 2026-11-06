"""Tests for Server-Sent Events (SSE) Streaming."""

from pathlib import Path

from fastapi.testclient import TestClient

from signalrag.api.app import app
from signalrag.api.streaming_sse import generate_sse_events


def test_generate_sse_events():
    tokens = ["Hello ", "world!"]
    citations = [{"index": 1, "source": "test.txt", "text": "citation text"}]
    events = list(generate_sse_events("test query", tokens, citations))

    assert len(events) == 5  # start, token 1, token 2, citation, done
    assert "start" in events[0]
    assert "Hello " in events[1]
    assert "world!" in events[2]
    assert "citation" in events[3]
    assert "done" in events[4]


def test_api_query_stream_endpoint():
    client = TestClient(app)
    response = client.post("/query/stream", json={"query": "test streaming", "top_k": 3})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    content = response.text
    assert "data: " in content
    assert '"event": "token"' in content
    assert '"event": "done"' in content


def test_go_stream_client_source():
    go_path = Path("clients/go/cmd/signalrag-stream/main.go")
    assert go_path.exists()
    assert "package main" in go_path.read_text()
