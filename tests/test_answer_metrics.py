"""Tests for generation metrics: faithfulness, relevance, and citation precision."""

from signalrag.evaluation.answer_metrics import (
    AnswerRelevanceEvaluator,
    CitationPrecisionEvaluator,
    FaithfulnessEvaluator,
    evaluate_answer,
)
from signalrag.models.citation import CitationSource


def test_faithfulness_evaluator_grounded():
    context = [
        "BM25 is a ranking function used by search engines to estimate the relevance of documents.",
        "Reciprocal rank fusion combines scores from multiple search models.",
    ]
    answer = (
        "BM25 is a ranking function used to estimate document relevance. "
        "Reciprocal rank fusion merges search rankings."
    )
    evaluator = FaithfulnessEvaluator(token_overlap_threshold=0.3)
    score, claims = evaluator.evaluate(answer, context)
    assert score == 1.0
    assert len(claims) == 2
    assert all(c.is_grounded for c in claims)


def test_faithfulness_evaluator_hallucinated():
    context = ["Python 3.11 introduced substantial performance enhancements and exception groups."]
    answer = "Quantum gravity was discovered by Isaac Newton using telescope arrays."
    evaluator = FaithfulnessEvaluator(token_overlap_threshold=0.3)
    score, claims = evaluator.evaluate(answer, context)
    assert score == 0.0
    assert len(claims) == 1
    assert not claims[0].is_grounded


def test_answer_relevance_evaluator():
    question = "How does HNSW graph indexing work?"
    ground_truth = (
        "HNSW builds hierarchical proximity graphs with fast logarithmic search complexity."
    )
    answer = "HNSW constructs hierarchical proximity graphs offering logarithmic nearest neighbor search."
    unrelated_answer = "Cooking pasta requires boiling water with a teaspoon of salt."

    evaluator = AnswerRelevanceEvaluator()
    rel_good = evaluator.evaluate(question, answer, ground_truth)
    rel_bad = evaluator.evaluate(question, unrelated_answer, ground_truth)
    assert rel_good > 0.4
    assert rel_bad < 0.2


def test_citation_precision():
    context = ["PostgreSQL supports JSON indexing with GIN indexes."]
    evaluator = CitationPrecisionEvaluator()

    valid_cit = [
        CitationSource(chunk_id="c1", document_id="d1", snippet="PostgreSQL supports GIN indexes")
    ]
    invalid_cit = [
        CitationSource(
            chunk_id="c2", document_id="d2", snippet="Submarines navigate through sonar echoes"
        )
    ]

    assert evaluator.evaluate(valid_cit, context) == 1.0
    assert evaluator.evaluate(invalid_cit, context) == 0.0


def test_evaluate_answer_integration():
    res = evaluate_answer(
        question="What is RRF?",
        generated_answer="RRF combines ranking results from BM25 and vector search.",
        context_chunks=["RRF combines ranking results from BM25 and vector search."],
        ground_truth="RRF combines BM25 and vector rankings.",
    )
    assert res.faithfulness == 1.0
    assert res.answer_relevance > 0.5
    assert res.hallucination_score == 0.0
