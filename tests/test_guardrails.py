"""Tests for grounded-answer guardrails."""

from signalrag.generation import (
    CitationTracker,
    GroundedAnswerGuardrail,
    GuardrailStatus,
)
from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult


def test_guardrails_pass_well_grounded_answer():
    guardrail = GroundedAnswerGuardrail()
    tracker = CitationTracker()

    c1 = Chunk.create("d1", 0, "SignalRAG uses hybrid search merging BM25 and vector scores.", "doc.md")
    r1 = SearchResult(chunk=c1, score=0.9, rank=1)

    answer = "SignalRAG uses hybrid search [1]."
    tracked = tracker.track(answer, [r1])

    eval_res = guardrail.evaluate("What search does SignalRAG use?", [r1], tracked)
    assert eval_res.status == GuardrailStatus.PASSED
    assert eval_res.groundedness_score >= 0.4
    assert eval_res.context_sufficiency_score > 0.3


def test_guardrails_refuses_when_context_empty_or_low_score():
    guardrail = GroundedAnswerGuardrail(min_context_score=0.2)
    tracker = CitationTracker()

    eval_res = guardrail.evaluate(
        query="What is the GDP of Atlantis?",
        results=[],
        tracked_answer=tracker.track("Atlantis GDP is 100 billion.", []),
    )
    assert eval_res.status == GuardrailStatus.REFUSED
    assert "cannot find sufficient evidence" in eval_res.sanitized_answer


def test_guardrails_detects_explicit_refusal():
    guardrail = GroundedAnswerGuardrail()
    tracker = CitationTracker()

    c1 = Chunk.create("d1", 0, "Random text about apples.", "doc.md")
    r1 = SearchResult(chunk=c1, score=0.8, rank=1)

    refusal_answer = "I cannot find sufficient evidence in the provided documents to answer this question."
    tracked = tracker.track(refusal_answer, [r1])

    eval_res = guardrail.evaluate("What is quantum computing?", [r1], tracked)
    assert eval_res.status == GuardrailStatus.REFUSED


def test_guardrails_qualifies_unsupported_claim():
    guardrail = GroundedAnswerGuardrail(min_groundedness_score=0.5)
    tracker = CitationTracker()

    c1 = Chunk.create("d1", 0, "SignalRAG is written in Python.", "doc.md")
    r1 = SearchResult(chunk=c1, score=0.8, rank=1)

    # Claim asserts something not in the chunk with citation [1]
    hallucinated_answer = "SignalRAG was written by astronauts on the International Space Station [1]."
    tracked = tracker.track(hallucinated_answer, [r1])

    eval_res = guardrail.evaluate("Who wrote SignalRAG?", [r1], tracked)
    assert eval_res.status in (GuardrailStatus.QUALIFIED, GuardrailStatus.REFUSED)
