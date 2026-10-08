"""Streaming responses and event emission for RAG generation."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class StreamEvent:
    """Event emitted during token streaming."""

    event_type: Literal["token", "citation", "guardrail", "complete", "trace"]
    data: Any

    def to_sse(self) -> str:
        """Format as Server-Sent Event (SSE) line."""
        payload = json.dumps({"type": self.event_type, "data": self.data})
        return f"data: {payload}\n\n"


class StreamingRAGResponse:
    """Wraps stream event generator and provides token iteration or full accumulation."""

    def __init__(self, event_stream: Iterator[StreamEvent]) -> None:
        self._stream = event_stream
        self._accumulated_tokens: list[str] = []
        self._completed_data: dict[str, Any] | None = None

    def __iter__(self) -> Iterator[StreamEvent]:
        for event in self._stream:
            if event.event_type == "token":
                self._accumulated_tokens.append(str(event.data))
            elif event.event_type == "complete":
                self._completed_data = event.data
            yield event

    def tokens(self) -> Iterator[str]:
        """Yield text tokens only."""
        for event in self:
            if event.event_type == "token":
                yield str(event.data)

    def to_sse(self) -> Iterator[str]:
        """Format stream as Server-Sent Events."""
        for event in self:
            yield event.to_sse()
        yield "data: [DONE]\n\n"

    def accumulate_text(self) -> str:
        """Exhaust stream and return complete assembled answer."""
        list(self.tokens())
        return "".join(self._accumulated_tokens)
