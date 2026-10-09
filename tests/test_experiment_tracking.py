"""Tests for ExperimentTracker storage and retrieval."""

from pathlib import Path

from signalrag.evaluation.retrieval_metrics import RetrievalMetricScores
from signalrag.evaluation.runner import EvaluationReport, LatencyStats
from signalrag.evaluation.tracker import ExperimentTracker


def test_experiment_tracker(tmp_path: Path):
    tracker = ExperimentTracker(storage_dir=tmp_path)
    assert len(tracker.list_runs()) == 0

    report = EvaluationReport(
        benchmark_name="test-run",
        total_examples=10,
        successful_examples=10,
        retrieval_metrics=RetrievalMetricScores(recall_at_5=0.88, mrr=0.81),
        avg_faithfulness=0.92,
        avg_answer_relevance=0.89,
        avg_citation_precision=0.95,
        latency=LatencyStats(avg_s=1.5),
    )

    run = tracker.log_run(
        name="baseline_hybrid",
        config_params={"chunk_size": 512, "retriever": "hybrid"},
        report=report,
        git_commit="abcdef1",
        run_id="exp-test-01",
    )

    assert run.id == "exp-test-01"
    assert len(tracker.list_runs()) == 1

    fetched = tracker.get_run("exp-test-01")
    assert fetched is not None
    assert fetched.name == "baseline_hybrid"
    assert fetched.report.retrieval_metrics.recall_at_5 == 0.88

    # Delete
    assert tracker.delete_run("exp-test-01") is True
    assert tracker.get_run("exp-test-01") is None
    assert tracker.delete_run("exp-test-01") is False
