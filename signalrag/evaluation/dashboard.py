"""Terminal dashboard renderer for SignalRAG evaluation metrics."""

from __future__ import annotations

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from signalrag.evaluation.runner import EvaluationReport


def render_evaluation_dashboard(report: EvaluationReport, console: Console | None = None) -> None:
    """Render rich console evaluation dashboard matching SignalRAG benchmark specifications."""
    c = console or Console()

    # Outer panel container
    retrieval_text = (
        f"[bold cyan]Recall@5[/bold cyan]             {report.retrieval_metrics.recall_at_5 * 100:.1f}%\n"
        f"[bold cyan]Recall@10[/bold cyan]            {report.retrieval_metrics.recall_at_10 * 100:.1f}%\n"
        f"[bold cyan]MRR[/bold cyan]                   {report.retrieval_metrics.mrr:.2f}\n"
        f"[bold cyan]NDCG@10[/bold cyan]               {report.retrieval_metrics.ndcg_at_10:.2f}"
    )

    generation_text = (
        f"[bold green]Faithfulness[/bold green]          {report.avg_faithfulness * 100:.1f}%\n"
        f"[bold green]Answer Relevance[/bold green]      {report.avg_answer_relevance * 100:.1f}%\n"
        f"[bold green]Citation Precision[/bold green]    {report.avg_citation_precision * 100:.1f}%"
    )

    perf_text = (
        f"[bold yellow]P50 latency[/bold yellow]           {report.latency.p50_s:.1f}s\n"
        f"[bold yellow]P95 latency[/bold yellow]           {report.latency.p95_s:.1f}s\n"
        f"[bold yellow]Avg latency[/bold yellow]           {report.latency.avg_s:.1f}s"
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Category", style="bold white")
    table.add_column("Metrics")

    table.add_row("Retrieval", retrieval_text)
    table.add_row("", "")
    table.add_row("Generation", generation_text)
    table.add_row("", "")
    table.add_row("Performance", perf_text)

    panel = Panel(
        table,
        title="[bold green]SignalRAG Evaluation Dashboard[/bold green]",
        subtitle=f"[dim]Evaluation: {report.total_examples} questions[/dim]",
        box=ROUNDED,
        expand=False,
    )
    c.print(panel)


def render_failures_table(report: EvaluationReport, console: Console | None = None) -> None:
    """Render table of failed or low-scoring query instances."""
    c = console or Console()
    failed = [r for r in report.results if not r.success or r.retrieval_scores.recall_at_5 < 0.5]
    if not failed:
        c.print("[green]✓ All evaluation questions met relevance thresholds![/green]")
        return

    table = Table(title=f"Failed / Low Retrieval Samples ({len(failed)})", box=ROUNDED)
    table.add_column("Example ID", style="cyan")
    table.add_column("Question", style="bold")
    table.add_column("Recall@5", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Status", justify="center")

    for f in failed[:10]:
        table.add_row(
            f.example_id,
            f.question[:50] + ("..." if len(f.question) > 50 else ""),
            f"{f.retrieval_scores.recall_at_5 * 100:.1f}%",
            f"{f.retrieval_scores.mrr:.2f}",
            "[green]Success[/green]" if f.success else f"[red]{f.error_message}[/red]",
        )
    c.print(table)
