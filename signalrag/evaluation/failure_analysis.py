"""Automated failure mode classification and error diagnostics for RAG benchmarks."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from signalrag.evaluation.runner import EvaluationReport, ExampleResult


class FailureCategory(StrEnum):
    """Taxonomy of root cause failure modes in RAG systems."""

    CORRECT = "Correct retrieval and generation"
    MISSING_RELEVANT_DOC = "Missing relevant document"
    POOR_CHUNK_BOUNDARY = "Poor chunk boundary"
    QUERY_AMBIGUITY = "Query ambiguity"
    RERANKER_ERROR = "Reranker error"
    HALLUCINATION = "Hallucination"
    CITATION_FAILURE = "Citation failure"


class CategorizedFailure(BaseModel):
    """Classified failure diagnosis for an individual benchmark question."""

    example_id: str
    question: str
    category: FailureCategory
    root_cause: str
    suggested_remediation: str


class FailureAnalysisSummary(BaseModel):
    """Aggregate taxonomy breakdown across all evaluated questions."""

    total_questions: int
    correct_count: int
    retrieval_failure_count: int
    generation_failure_count: int
    citation_failure_count: int
    failure_mode_counts: dict[str, int] = Field(default_factory=dict)
    failures: list[CategorizedFailure] = Field(default_factory=list)


class FailureAnalyzer:
    """Classifies benchmark query errors into specific algorithmic failure categories."""

    def analyze(self, report: EvaluationReport) -> FailureAnalysisSummary:
        """Classify each query result in an evaluation report."""
        correct = 0
        ret_fails = 0
        gen_fails = 0
        cit_fails = 0
        category_counts: dict[str, int] = {}
        failure_list: list[CategorizedFailure] = []

        for item in report.results:
            cat, cause, fix = self._classify_example(item)
            if cat == FailureCategory.CORRECT:
                correct += 1
            else:
                if cat in (
                    FailureCategory.MISSING_RELEVANT_DOC,
                    FailureCategory.POOR_CHUNK_BOUNDARY,
                    FailureCategory.QUERY_AMBIGUITY,
                    FailureCategory.RERANKER_ERROR,
                ):
                    ret_fails += 1
                elif cat == FailureCategory.HALLUCINATION:
                    gen_fails += 1
                elif cat == FailureCategory.CITATION_FAILURE:
                    cit_fails += 1

                category_counts[cat.value] = category_counts.get(cat.value, 0) + 1
                failure_list.append(
                    CategorizedFailure(
                        example_id=item.example_id,
                        question=item.question,
                        category=cat,
                        root_cause=cause,
                        suggested_remediation=fix,
                    )
                )

        return FailureAnalysisSummary(
            total_questions=report.total_examples,
            correct_count=correct,
            retrieval_failure_count=ret_fails,
            generation_failure_count=gen_fails,
            citation_failure_count=cit_fails,
            failure_mode_counts=dict(
                sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            ),
            failures=failure_list,
        )

    def _classify_example(self, item: ExampleResult) -> tuple[FailureCategory, str, str]:
        """Determine failure taxonomy based on multi-stage signals."""
        if not item.success:
            return (
                FailureCategory.MISSING_RELEVANT_DOC,
                f"Execution error: {item.error_message}",
                "Verify database connectivity and query syntax.",
            )

        # Retrieval metrics check
        recall5 = item.retrieval_scores.recall_at_5
        has_retrieval_hit = recall5 > 0.0

        if not has_retrieval_hit:
            # Did it fail due to query ambiguity or completely missing doc?
            words = item.question.split()
            if len(words) <= 3:
                return (
                    FailureCategory.QUERY_AMBIGUITY,
                    "Query is too short or ambiguous to establish distinctive lexical/semantic matches.",
                    "Enhance query rewriter with multi-query expansion and acronym lookup.",
                )
            if item.retrieval_scores.recall_at_10 > 0.0:
                return (
                    FailureCategory.RERANKER_ERROR,
                    "Target chunk was retrieved in top-10 but demoted by reranker outside top-5.",
                    "Tune cross-encoder temperature or linear hybrid weight balance.",
                )
            return (
                FailureCategory.MISSING_RELEVANT_DOC,
                "Neither BM25 nor dense vector retriever retrieved expected document.",
                "Verify corpus coverage or lower vector similarity threshold.",
            )

        # Retrieval passed; check generation
        if item.generation_scores:
            if item.generation_scores.faithfulness < 0.6:
                return (
                    FailureCategory.HALLUCINATION,
                    f"Generated claims contradicted retrieved context (faithfulness: {item.generation_scores.faithfulness:.2f}).",
                    "Strengthen LLM grounding prompt and lower temperature.",
                )
            if item.generation_scores.citation_precision < 0.5:
                return (
                    FailureCategory.CITATION_FAILURE,
                    "Generated answer claims support from irrelevant citation chunks.",
                    "Tighten citation extraction parser and verify sentence-to-chunk binding.",
                )

        return (FailureCategory.CORRECT, "All benchmarks passed.", "None")


def render_failure_analysis(
    summary: FailureAnalysisSummary, console: Console | None = None
) -> None:
    """Render Rich terminal failure categorization tables."""
    c = console or Console()

    overview_table = Table(box=ROUNDED, show_header=False)
    overview_table.add_column("Category")
    overview_table.add_column("Count", justify="right")

    overview_table.add_row(
        "[green]✓ Correct retrieval & generation[/green]",
        f"[bold green]{summary.correct_count}[/bold green]",
    )
    overview_table.add_row(
        "[red]✗ Retrieval failure[/red]",
        f"[bold red]{summary.retrieval_failure_count}[/bold red]",
    )
    overview_table.add_row(
        "[red]✗ Generation failure[/red]",
        f"[bold red]{summary.generation_failure_count}[/bold red]",
    )
    overview_table.add_row(
        "[red]✗ Citation failure[/red]",
        f"[bold red]{summary.citation_failure_count}[/bold red]",
    )

    c.print(
        Panel(
            overview_table,
            title=f"[bold cyan]Failure Categorization ({summary.total_questions} questions)[/bold cyan]",
            box=ROUNDED,
        )
    )

    if summary.failure_mode_counts:
        modes_table = Table(title="Top Failure Modes", box=ROUNDED)
        modes_table.add_column("Rank", justify="center")
        modes_table.add_column("Failure Mode", style="bold")
        modes_table.add_column("Count", justify="right", style="cyan")

        for idx, (mode, count) in enumerate(summary.failure_mode_counts.items(), start=1):
            modes_table.add_row(str(idx), mode, str(count))
        c.print(modes_table)
