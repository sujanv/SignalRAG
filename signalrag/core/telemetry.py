"""Prometheus metrics exposition and distributed tracing abstractions."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Generator
from typing import Any


class TelemetryCollector:
    """Collects system telemetry and formats Prometheus exposition metrics."""

    def __init__(self):
        self.queries_total: dict[str, int] = {"success": 0, "error": 0}
        self.total_latency_seconds: float = 0.0
        self.query_count: int = 0
        self.retrieved_chunks_total: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def record_query(self, latency_seconds: float, status: str = "success") -> None:
        self.queries_total[status] = self.queries_total.get(status, 0) + 1
        self.total_latency_seconds += latency_seconds
        self.query_count += 1

    def record_retrieval(self, chunk_count: int) -> None:
        self.retrieved_chunks_total += chunk_count

    def record_cache_event(self, hit: bool) -> None:
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    @contextlib.contextmanager
    def trace_span(self, span_name: str) -> Generator[dict[str, Any], None, None]:
        """Lightweight OpenTelemetry compatible span context manager."""
        t0 = time.perf_counter()
        span_data = {"name": span_name, "start_time": t0, "status": "OK"}
        try:
            yield span_data
        except Exception as exc:
            span_data["status"] = "ERROR"
            span_data["error"] = str(exc)
            raise
        finally:
            span_data["duration_ms"] = (time.perf_counter() - t0) * 1000

    def format_prometheus(self) -> str:
        """Render metrics in Prometheus text exposition format."""
        lines = [
            "# HELP signalrag_queries_total Total number of RAG queries executed.",
            "# TYPE signalrag_queries_total counter",
            f'signalrag_queries_total{{status="success"}} {self.queries_total.get("success", 0)}',
            f'signalrag_queries_total{{status="error"}} {self.queries_total.get("error", 0)}',
            "",
            "# HELP signalrag_latency_seconds_sum Total cumulative latency across queries.",
            "# TYPE signalrag_latency_seconds_sum counter",
            f"signalrag_latency_seconds_sum {self.total_latency_seconds:.6f}",
            "",
            "# HELP signalrag_latency_seconds_count Total count of latency measurements.",
            "# TYPE signalrag_latency_seconds_count counter",
            f"signalrag_latency_seconds_count {self.query_count}",
            "",
            "# HELP signalrag_retrieved_chunks_total Total number of candidate chunks retrieved.",
            "# TYPE signalrag_retrieved_chunks_total counter",
            f"signalrag_retrieved_chunks_total {self.retrieved_chunks_total}",
            "",
            "# HELP signalrag_cache_hits_total Total number of retrieval cache hits.",
            "# TYPE signalrag_cache_hits_total counter",
            f"signalrag_cache_hits_total {self.cache_hits}",
            "",
            "# HELP signalrag_cache_misses_total Total number of retrieval cache misses.",
            "# TYPE signalrag_cache_misses_total counter",
            f"signalrag_cache_misses_total {self.cache_misses}",
            "",
        ]
        return "\n".join(lines)


global_telemetry = TelemetryCollector()
