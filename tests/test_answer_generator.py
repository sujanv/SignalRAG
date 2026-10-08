"""Tests for RAG answer generation."""

from signalrag.generation import AnswerGenerator, MockLLMClient, format_rag_prompt
from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult


def test_format_rag_prompt():
    c1 = Chunk.create("d1", 0, "SignalRAG uses hybrid search.", "docs/rag.md", page_number=3)
    res = SearchResult(chunk=c1, score=0.9, rank=1)

    sys_prompt, user_prompt = format_rag_prompt("How does SignalRAG search?", [res])
    assert "SignalRAG" in sys_prompt
    assert "Reference Documents:" in user_prompt
    assert "[1] Source: docs/rag.md, Page 3" in user_prompt
    assert "SignalRAG uses hybrid search." in user_prompt
    assert "Question: How does SignalRAG search?" in user_prompt


def test_answer_generator_grounded_response():
    c1 = Chunk.create(
        "d1",
        0,
        "Mean Reciprocal Rank (MRR) evaluates the position of the first relevant retrieved document.",
        "eval.md",
    )
    res = SearchResult(chunk=c1, score=0.95, rank=1)

    generator = AnswerGenerator(llm_client=MockLLMClient())
    answer_obj = generator.generate("What is Mean Reciprocal Rank MRR?", [res])

    assert answer_obj.answer is not None
    assert len(answer_obj.answer) > 0
    assert "[1]" in answer_obj.answer
    assert "Mean Reciprocal Rank" in answer_obj.answer
    assert answer_obj.duration_seconds >= 0.0


def test_answer_generator_refusal_when_no_evidence():
    c1 = Chunk.create("d1", 0, "Deep space exploration missions to Jupiter and Saturn.", "space.md")
    res = SearchResult(chunk=c1, score=0.1, rank=1)

    generator = AnswerGenerator(llm_client=MockLLMClient())
    answer_obj = generator.generate("What is the recipe for chocolate cake?", [res])

    assert "I cannot find sufficient evidence" in answer_obj.answer
