"""Tests for Experiment comparison."""

from rich.console import Console

from signalrag.evaluation.comparison import compare_experiments, render_comparison_table
from signalrag.evaluation.retrieval_metrics import RetrievalMetricScores
from signalrag.evaluation.runner import EvaluationReport, LatencyStats
from signalrag.evaluation.tracker import ExperimentRun


def test_compare_experiments():
    report_a = EvaluationReport(
        benchmark_name="test",
        total_examples=10,
        successful_examples=10,
        retrieval_metrics=RetrievalMetricScores(recall_at_5=0.842, mrr=0.74),
        avg_faithfulness=0.881,
        avg_answer_relevance=0.85,
        avg_citation_precision=0.90,
        latency=LatencyStats(avg_s=1.4),
    )
    report_b = EvaluationReport(
        benchmark_name="test",
        total_examples=10,
        successful_examples=10,
        retrieval_metrics=RetrievalMetricScores(recall_at_5=0.897, mrr=0.83),
        avg_faithfulness=0.924,
        avg_answer_relevance=0.91,
        avg_citation_precision=0.95,
        latency=LatencyStats(avg_s=1.9),
    )

    run_a = ExperimentRun(
        id="exp-a",
        name="Experiment A",
        config_params={"chunk_size": 512, "retriever": "Hybrid"},
        report=report_a,
    )
    run_b = ExperimentRun(
        id="exp-b",
        name="Experiment B",
        config_params={"chunk_size": 768, "retriever": "Semantic"},
        report=report_b,
    )

    comp = compare_experiments(run_a, run_b)
    assert comp.run_a_name == "Experiment A"
    assert comp.params_diff["chunk_size"] == (512, 768)

    rec5_delta = next(m for m in comp.metric_deltas if m.metric_name == "Recall@5")
    assert rec5_delta.val_a == 0.842
    assert rec5_delta.val_b == 0.897
    assert rec5_delta.is_improved is True

    console = Console(record=True)
    render_comparison_table(comp, console=console)
    output = console.export_text()
    assert "Experiment Comparison: Experiment A vs Experiment B" in output
    assert "Recall@5" in output
