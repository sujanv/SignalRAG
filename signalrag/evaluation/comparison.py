"""Side-by-side experiment comparison and parameter/metric delta analysis."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table

from signalrag.evaluation.tracker import ExperimentRun


class MetricDelta(BaseModel):
    """Metric comparison row with delta calculation."""

    metric_name: str
    val_a: float
    val_b: float
    diff: float
    pct_change: float
    higher_is_better: bool = True

    @property
    def is_improved(self) -> bool:
        if self.higher_is_better:
            return self.diff > 0.0001
        return self.diff < -0.0001


class ExperimentComparison(BaseModel):
    """Complete comparative evaluation between two experiment runs."""

    run_a_id: str
    run_a_name: str
    run_b_id: str
    run_b_name: str
    params_diff: dict[str, tuple[Any, Any]] = Field(default_factory=dict)
    metric_deltas: list[MetricDelta] = Field(default_factory=list)


def compare_experiments(run_a: ExperimentRun, run_b: ExperimentRun) -> ExperimentComparison:
    """Compare two experiment runs across configuration parameters and evaluation metrics."""
    # 1. Compare parameters
    all_keys = set(run_a.config_params.keys()) | set(run_b.config_params.keys())
    params_diff = {}
    for k in sorted(all_keys):
        val_a = run_a.config_params.get(k, "—")
        val_b = run_b.config_params.get(k, "—")
        params_diff[k] = (val_a, val_b)

    # 2. Compare metrics
    m_a = run_a.report.retrieval_metrics
    m_b = run_b.report.retrieval_metrics

    metrics_to_compare = [
        ("Recall@5", m_a.recall_at_5, m_b.recall_at_5, True),
        ("Recall@10", m_a.recall_at_10, m_b.recall_at_10, True),
        ("MRR", m_a.mrr, m_b.mrr, True),
        ("NDCG@10", m_a.ndcg_at_10, m_b.ndcg_at_10, True),
        ("Faithfulness", run_a.report.avg_faithfulness, run_b.report.avg_faithfulness, True),
        (
            "Answer Relevance",
            run_a.report.avg_answer_relevance,
            run_b.report.avg_answer_relevance,
            True,
        ),
        ("Avg Latency (s)", run_a.report.latency.avg_s, run_b.report.latency.avg_s, False),
    ]

    deltas = []
    for name, v_a, v_b, higher_better in metrics_to_compare:
        diff = v_b - v_a
        pct = (diff / v_a * 100.0) if v_a != 0 else 0.0
        deltas.append(
            MetricDelta(
                metric_name=name,
                val_a=round(v_a, 4),
                val_b=round(v_b, 4),
                diff=round(diff, 4),
                pct_change=round(pct, 2),
                higher_is_better=higher_better,
            )
        )

    return ExperimentComparison(
        run_a_id=run_a.id,
        run_a_name=run_a.name,
        run_b_id=run_b.id,
        run_b_name=run_b.name,
        params_diff=params_diff,
        metric_deltas=deltas,
    )


def render_comparison_table(
    comparison: ExperimentComparison, console: Console | None = None
) -> None:
    """Render side-by-side terminal comparison table."""
    c = console or Console()

    table = Table(
        box=ROUNDED,
        title=f"Experiment Comparison: {comparison.run_a_name} vs {comparison.run_b_name}",
    )
    table.add_column("Parameter / Metric", style="bold")
    table.add_column(f"{comparison.run_a_name} ({comparison.run_a_id})", justify="center")
    table.add_column(f"{comparison.run_b_name} ({comparison.run_b_id})", justify="center")
    table.add_column("Delta", justify="right")

    # Parameters section
    if comparison.params_diff:
        table.add_section()
        for param, (val_a, val_b) in comparison.params_diff.items():
            table.add_row(f"[dim]{param}[/dim]", str(val_a), str(val_b), "")

    # Metrics section
    table.add_section()
    for m in comparison.metric_deltas:
        if "s" in m.metric_name:
            str_a = f"{m.val_a:.2f}s"
            str_b = f"{m.val_b:.2f}s"
            delta_str = f"{m.diff:+.2f}s"
        elif (
            "Recall" in m.metric_name
            or "Faithfulness" in m.metric_name
            or "Relevance" in m.metric_name
        ):
            str_a = f"{m.val_a * 100:.1f}%"
            str_b = f"{m.val_b * 100:.1f}%"
            delta_str = f"{m.diff * 100:+.1f}%"
        else:
            str_a = f"{m.val_a:.2f}"
            str_b = f"{m.val_b:.2f}"
            delta_str = f"{m.diff:+.2f}"

        if abs(m.diff) < 0.0001:
            colored_delta = f"[dim]{delta_str}[/dim]"
        elif m.is_improved:
            colored_delta = f"[bold green]{delta_str} (▲)[/bold green]"
        else:
            colored_delta = f"[bold red]{delta_str} (▼)[/bold red]"

        table.add_row(m.metric_name, str_a, str_b, colored_delta)

    c.print(table)
