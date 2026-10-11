"""Tests for Prometheus metrics and telemetry tracing."""

from fastapi.testclient import TestClient

from signalrag.api.app import app
from signalrag.core.telemetry import TelemetryCollector, global_telemetry


def test_telemetry_collector():
    collector = TelemetryCollector()
    collector.record_query(0.045, status="success")
    collector.record_retrieval(5)
    collector.record_cache_event(hit=True)
    collector.record_cache_event(hit=False)

    prom_text = collector.format_prometheus()
    assert 'signalrag_queries_total{status="success"} 1' in prom_text
    assert "signalrag_retrieved_chunks_total 5" in prom_text
    assert "signalrag_cache_hits_total 1" in prom_text
    assert "signalrag_cache_misses_total 1" in prom_text


def test_telemetry_tracing_context():
    collector = TelemetryCollector()
    with collector.trace_span("test_span") as span:
        assert span["name"] == "test_span"
        assert span["status"] == "OK"
    assert "duration_ms" in span


def test_prometheus_api_endpoint():
    client = TestClient(app)
    global_telemetry.record_query(0.012, status="success")
    res = client.get("/metrics/prometheus")
    assert res.status_code == 200
    assert "signalrag_queries_total" in res.text
