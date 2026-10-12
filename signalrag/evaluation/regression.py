"""Automated RAG regression evaluation gatekeeper."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MetricDelta(BaseModel):
    """Comparison of a single metric between baseline and candidate."""

    metric_name: str
    baseline_value: float
    candidate_value: float
    diff: float
    is_regression: bool


class RegressionCheckResult(BaseModel):
    """Comprehensive evaluation regression check report."""

    passed: bool
    max_allowed_drop: float
    deltas: list[MetricDelta] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)


class RegressionGuard:
    """Evaluates candidate performance against baseline metrics to prevent quality degradation."""

    CRITICAL_METRICS = [
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "faithfulness",
        "answer_relevance",
    ]

    def __init__(self, max_allowed_drop: float = 0.05):
        self.max_allowed_drop = max_allowed_drop

    def check(
        self,
        baseline_metrics: dict[str, Any],
        candidate_metrics: dict[str, Any],
    ) -> RegressionCheckResult:
        """Compare candidate against baseline metrics. Flags regressions exceeding max_allowed_drop."""
        deltas: list[MetricDelta] = []
        regressions: list[str] = []

        # Flatten nested metric dicts if necessary
        base_flat = self._flatten(baseline_metrics)
        cand_flat = self._flatten(candidate_metrics)

        for metric in self.CRITICAL_METRICS:
            if metric in base_flat and metric in cand_flat:
                b_val = float(base_flat[metric])
                c_val = float(cand_flat[metric])
                diff = c_val - b_val

                # For quality metrics, a negative diff represents a degradation
                is_regr = diff < -self.max_allowed_drop
                if is_regr:
                    regressions.append(metric)

                deltas.append(
                    MetricDelta(
                        metric_name=metric,
                        baseline_value=round(b_val, 4),
                        candidate_value=round(c_val, 4),
                        diff=round(diff, 4),
                        is_regression=is_regr,
                    )
                )

        passed = len(regressions) == 0
        return RegressionCheckResult(
            passed=passed,
            max_allowed_drop=self.max_allowed_drop,
            deltas=deltas,
            regressions=regressions,
        )

    def _flatten(self, d: dict[str, Any]) -> dict[str, Any]:
        flat = {}
        for k, v in d.items():
            if isinstance(v, dict):
                flat.update(self._flatten(v))
            else:
                flat[k] = v
        return flat
