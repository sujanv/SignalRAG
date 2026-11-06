"""Server-Sent Events (SSE) streaming generator for real-time RAG response tokens and citations."""

from __future__ import annotations

import json
import time
from collections.abc import Generator


def generate_sse_events(
    query: str,
    answer_tokens: list[str],
    citations: list[dict[str, str]] | None = None,
) -> Generator[str, None, None]:
    """Yield formatted SSE event strings for progressive frontend rendering."""
    t0 = time.perf_counter()

    # 1. Start event
    yield f"data: {json.dumps({'event': 'start', 'query': query})}\n\n"

    # 2. Token events
    for token in answer_tokens:
        yield f"data: {json.dumps({'event': 'token', 'content': token})}\n\n"

    # 3. Citation events
    if citations:
        for cit in citations:
            yield f"data: {json.dumps({'event': 'citation', 'data': cit})}\n\n"

    # 4. Completed event
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    yield f"data: {json.dumps({'event': 'done', 'latency_ms': elapsed_ms})}\n\n"
