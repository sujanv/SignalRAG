"""Tests for Observability tracking and metrics."""

import time

from signalrag.core.observability import MetricsCollector, QueryTrace


def test_query_trace_spans():
    trace = QueryTrace(query="What is BM25?")
    trace.add_span("embedding", 12.5, {"model": "hash"})
    trace.add_span("retrieval", 25.1, {"top_k": 5})
    trace.prompt_tokens = 150
    trace.completion_tokens = 50
    trace.total_tokens = 200
    trace.total_latency_ms = 40.0

    assert len(trace.spans) == 2
    assert trace.spans[0].name == "embedding"
    assert trace.total_tokens == 200


def test_metrics_collector():
    collector = MetricsCollector()
    trace = QueryTrace(query="Hello")
    with collector.span("retrieval", trace):
        time.sleep(0.01)

    trace.total_latency_ms = 15.0
    trace.prompt_tokens = 100
    trace.completion_tokens = 30
    collector.record_query(trace)

    summary = collector.get_summary()
    assert summary["total_queries"] == 1
    assert summary["total_tokens"] == 130
    assert summary["avg_retrieval_latency_ms"] > 0.0

    collector.reset()
    assert collector.get_summary()["total_queries"] == 0
