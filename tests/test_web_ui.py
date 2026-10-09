"""Tests for SignalRAG Web UI static serving."""

from fastapi.testclient import TestClient

from signalrag.api.app import app

client = TestClient(app)


def test_web_ui_endpoints():
    res = client.get("/")
    assert res.status_code == 200
    assert "SignalRAG" in res.text
    assert "Ask a question..." in res.text
    assert "Sources & Citations" in res.text

    res_ui = client.get("/ui")
    assert res_ui.status_code == 200
    assert "SignalRAG" in res_ui.text
