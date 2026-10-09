"""Observability framework tracking stage latencies, token consumption, and metrics."""

from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, Field


class PipelineSpan(BaseModel):
    """Execution span capturing timing and context for an isolated stage."""

    name: str
    duration_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryTrace(BaseModel):
    """Complete diagnostic trace for a single user query."""

    trace_id: str = Field(default_factory=lambda: f"tr-{uuid.uuid4().hex[:8]}")
    query: str
    spans: list[PipelineSpan] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add_span(
        self, name: str, duration_ms: float, metadata: dict[str, Any] | None = None
    ) -> None:
        self.spans.append(
            PipelineSpan(name=name, duration_ms=round(duration_ms, 2), metadata=metadata or {})
        )


class MetricsCollector:
    """Accumulates operational metrics across queries."""

    def __init__(self):
        self._total_queries: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._stage_latencies: dict[str, list[float]] = {
            "retrieval": [],
            "embedding": [],
            "reranking": [],
            "llm": [],
            "total": [],
        }

    def record_query(self, trace: QueryTrace) -> None:
        """Record completed trace into aggregate metrics."""
        self._total_queries += 1
        self._total_prompt_tokens += trace.prompt_tokens
        self._total_completion_tokens += trace.completion_tokens
        self._stage_latencies["total"].append(trace.total_latency_ms)

        for span in trace.spans:
            if span.name in self._stage_latencies:
                self._stage_latencies[span.name].append(span.duration_ms)

    @contextlib.contextmanager
    def span(
        self, name: str, trace: QueryTrace | None = None, metadata: dict[str, Any] | None = None
    ) -> Generator[None, None, None]:
        """Context manager to measure latency of an execution span."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dur_ms = (time.perf_counter() - t0) * 1000
            if trace:
                trace.add_span(name, dur_ms, metadata)
            if name in self._stage_latencies:
                self._stage_latencies[name].append(dur_ms)

    def get_summary(self) -> dict[str, Any]:
        """Generate summary statistics for operational monitoring."""
        stage_avgs = {}
        for stage, times in self._stage_latencies.items():
            stage_avgs[f"avg_{stage}_latency_ms"] = (
                round(sum(times) / len(times), 2) if times else 0.0
            )

        return {
            "total_queries": self._total_queries,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            **stage_avgs,
        }

    def reset(self) -> None:
        """Reset all counters."""
        self._total_queries = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        for k in self._stage_latencies:
            self._stage_latencies[k].clear()


# Global default collector
global_metrics = MetricsCollector()
