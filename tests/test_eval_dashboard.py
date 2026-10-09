"""Tests for Evaluation Dashboard rendering."""

from rich.console import Console

from signalrag.evaluation.dashboard import render_evaluation_dashboard, render_failures_table
from signalrag.evaluation.retrieval_metrics import RetrievalMetricScores
from signalrag.evaluation.runner import EvaluationReport, ExampleResult, LatencyStats


def test_render_dashboard():
    report = EvaluationReport(
        benchmark_name="sample-bench",
        total_examples=10,
        successful_examples=9,
        retrieval_metrics=RetrievalMetricScores(
            recall_at_5=0.874,
            recall_at_10=0.931,
            mrr=0.81,
            ndcg_at_10=0.86,
        ),
        avg_faithfulness=0.912,
        avg_answer_relevance=0.897,
        avg_citation_precision=0.95,
        latency=LatencyStats(p50_s=1.2, p95_s=2.8, avg_s=1.8),
        results=[
            ExampleResult(
                example_id="q1",
                question="Why is X happening?",
                retrieval_scores=RetrievalMetricScores(recall_at_5=0.0, mrr=0.0),
                success=True,
            )
        ],
    )
    console = Console(record=True)
    render_evaluation_dashboard(report, console=console)
    output = console.export_text()
    assert "SignalRAG Evaluation Dashboard" in output
    assert "87.4%" in output
    assert "91.2%" in output
    assert "1.2s" in output

    render_failures_table(report, console=console)
    fail_out = console.export_text()
    assert "Failed / Low Retrieval Samples" in fail_out
