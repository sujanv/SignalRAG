"""Tests for token streaming and Server-Sent Events (SSE)."""

from signalrag.generation.streaming import StreamEvent, StreamingRAGResponse


def test_streaming_rag_response_tokens_and_accumulation():
    def dummy_events():
        yield StreamEvent(event_type="token", data="SignalRAG ")
        yield StreamEvent(event_type="token", data="streams ")
        yield StreamEvent(event_type="token", data="answers.")
        yield StreamEvent(event_type="citation", data=[{"index": 1, "source": "doc.md"}])
        yield StreamEvent(event_type="complete", data={"status": "passed"})

    stream = StreamingRAGResponse(dummy_events())
    tokens = list(stream.tokens())
    assert tokens == ["SignalRAG ", "streams ", "answers."]

    # Verify complete text
    assert stream.accumulate_text() == "SignalRAG streams answers."


def test_streaming_sse_format():
    def dummy_events():
        yield StreamEvent(event_type="token", data="word")

    stream = StreamingRAGResponse(dummy_events())
    sse_lines = list(stream.to_sse())

    assert len(sse_lines) == 2
    assert sse_lines[0].startswith("data: {")
    assert '"type": "token"' in sse_lines[0]
    assert sse_lines[1] == "data: [DONE]\n\n"
