"""Tests for RAG evaluation regression gatekeeper."""

from signalrag.evaluation.regression import RegressionGuard


def test_regression_guard_pass():
    guard = RegressionGuard(max_allowed_drop=0.03)
    baseline = {"recall_at_5": 0.85, "mrr": 0.80, "faithfulness": 0.90}
    candidate = {"recall_at_5": 0.86, "mrr": 0.79, "faithfulness": 0.92}

    res = guard.check(baseline, candidate)
    assert res.passed is True
    assert len(res.regressions) == 0


def test_regression_guard_fail():
    guard = RegressionGuard(max_allowed_drop=0.02)
    baseline = {"recall_at_5": 0.85, "mrr": 0.80}
    # Severe drop in recall
    candidate = {"recall_at_5": 0.78, "mrr": 0.80}

    res = guard.check(baseline, candidate)
    assert res.passed is False
    assert "recall_at_5" in res.regressions
