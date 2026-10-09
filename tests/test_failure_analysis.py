"""Tests for FailureAnalyzer categorization."""

from rich.console import Console

from signalrag.evaluation.answer_metrics import AnswerEvaluationResult
from signalrag.evaluation.failure_analysis import (
    FailureAnalyzer,
    render_failure_analysis,
)
from signalrag.evaluation.retrieval_metrics import RetrievalMetricScores
from signalrag.evaluation.runner import EvaluationReport, ExampleResult, LatencyStats


def test_failure_analyzer_categorization():
    report = EvaluationReport(
        benchmark_name="test",
        total_examples=4,
        successful_examples=4,
        retrieval_metrics=RetrievalMetricScores(),
        latency=LatencyStats(),
        results=[
            ExampleResult(
                example_id="q1",
                question="What is RRF?",
                retrieval_scores=RetrievalMetricScores(recall_at_5=1.0),
                generation_scores=AnswerEvaluationResult(
                    faithfulness=1.0, answer_relevance=1.0, citation_precision=1.0
                ),
                success=True,
            ),
            ExampleResult(
                example_id="q2",
                question="Why?",
                retrieval_scores=RetrievalMetricScores(recall_at_5=0.0, recall_at_10=0.0),
                success=True,
            ),
            ExampleResult(
                example_id="q3",
                question="How does indexing operate in signalrag?",
                retrieval_scores=RetrievalMetricScores(recall_at_5=0.0, recall_at_10=1.0),
                success=True,
            ),
            ExampleResult(
                example_id="q4",
                question="Explain query rewriting process?",
                retrieval_scores=RetrievalMetricScores(recall_at_5=1.0),
                generation_scores=AnswerEvaluationResult(
                    faithfulness=0.2, answer_relevance=0.8, citation_precision=0.9
                ),
                success=True,
            ),
        ],
    )

    analyzer = FailureAnalyzer()
    summary = analyzer.analyze(report)

    assert summary.total_questions == 4
    assert summary.correct_count == 1
    assert summary.retrieval_failure_count == 2
    assert summary.generation_failure_count == 1

    console = Console(record=True)
    render_failure_analysis(summary, console=console)
    output = console.export_text()
    assert "Failure Categorization" in output
    assert "Top Failure Modes" in output
