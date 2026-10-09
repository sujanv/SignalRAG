"""Main entry point for SignalRAG CLI."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from signalrag import __version__
from signalrag.indexing import IncrementalIndexer, IndexingPipeline, MemoryVectorStore
from signalrag.ingestion import DirectoryLoader, PDFLoader, TextLoader

app = typer.Typer(
    name="signalrag",
    help="SignalRAG: Production-grade RAG pipeline with retrieval and answer evaluation.",
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the version of SignalRAG."""
    console.print(f"[bold cyan]SignalRAG[/bold cyan] version [green]{__version__}[/green]")


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Path to document file or directory to ingest.")],
    recursive: Annotated[
        bool, typer.Option("--recursive", "-r", help="Recursively search directory.")
    ] = True,
    page_mode: Annotated[
        bool, typer.Option("--page-mode", help="Split PDFs into individual page documents.")
    ] = False,
    silent_errors: Annotated[
        bool, typer.Option("--silent-errors", help="Skip corrupted or unreadable files.")
    ] = False,
) -> None:
    """Ingest documents from a file or directory and produce structured documents."""
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] Path '{path}' does not exist.")
        raise typer.Exit(code=1)

    custom_loaders = None
    if page_mode:
        custom_loaders = {".pdf": lambda p: PDFLoader(p, mode="page")}

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PDFLoader(path, mode="page" if page_mode else "document")
        else:
            loader = TextLoader(path)
        docs = loader.load()
    else:
        loader = DirectoryLoader(
            path,
            recursive=recursive,
            custom_loaders=custom_loaders,
            silent_errors=silent_errors,
        )
        docs = loader.load()

    table = Table(title=f"Ingested Documents ({len(docs)} total)")
    table.add_column("Doc ID", style="cyan", no_wrap=True)
    table.add_column("File Name", style="magenta")
    table.add_column("Type", style="green")
    table.add_column("Pages", justify="right")
    table.add_column("Chars", justify="right")
    table.add_column("Title / Snippet", style="yellow")

    for doc in docs:
        pages = str(doc.metadata.total_pages or 1)
        chars = str(len(doc.text))
        snippet = doc.metadata.title or (
            doc.text[:40].replace("\n", " ") + "..." if doc.text else ""
        )
        table.add_row(
            doc.id[:10], doc.metadata.file_name, doc.metadata.file_type, pages, chars, snippet
        )

    console.print(table)
    console.print(
        f"[bold green]Successfully ingested {len(docs)} structured documents.[/bold green]"
    )


@app.command()
def index(
    path: Annotated[Path, typer.Argument(help="Path to documents to index.")],
    recursive: Annotated[
        bool, typer.Option("--recursive", "-r", help="Recursively search directory.")
    ] = True,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force reindexing of all documents.")
    ] = False,
    storage_dir: Annotated[
        Path, typer.Option("--storage-dir", help="Directory to store vector index.")
    ] = Path("./storage/vector_store"),
) -> None:
    """Incrementally parse, chunk, embed, and index documents."""
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] Path '{path}' does not exist.")
        raise typer.Exit(code=1)

    index_path = storage_dir / "index.json"
    ledger_path = storage_dir / "ledger.json"

    vector_store = MemoryVectorStore(storage_path=index_path)
    if index_path.exists():
        vector_store.load(index_path)

    pipeline = IndexingPipeline(vector_store=vector_store)
    indexer = IncrementalIndexer(pipeline=pipeline, ledger_path=ledger_path)

    # Ingest documents
    if path.is_file():
        loader = PDFLoader(path) if path.suffix.lower() == ".pdf" else TextLoader(path)
        docs = loader.load()
    else:
        loader = DirectoryLoader(path, recursive=recursive)
        docs = loader.load()

    result = indexer.index(docs, force=force)

    table = Table(title="SignalRAG Incremental Indexing Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Total Examined", str(result.total_examined))
    table.add_row("Newly Added Docs", str(result.added_documents))
    table.add_row("Updated Docs", str(result.updated_documents))
    table.add_row("Skipped (Unchanged)", str(result.skipped_documents))
    table.add_row("New Chunks Indexed", str(result.added_chunks))
    table.add_row("Total Store Chunks", str(result.stats.get("vector_store_chunks", 0)))
    table.add_row("Duration", f"{result.duration_seconds}s")

    console.print(table)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query text.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of results to retrieve.")] = 5,
    hybrid: Annotated[
        bool, typer.Option("--hybrid/--vector-only", help="Use hybrid search (BM25 + Semantic).")
    ] = True,
    rerank: Annotated[bool, typer.Option("--rerank", help="Apply second-stage reranking.")] = False,
    compress: Annotated[
        bool, typer.Option("--compress", help="Compress retrieved context chunks.")
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", "-d", help="Display diagnostic pipeline trace.")
    ] = False,
    author: Annotated[str | None, typer.Option("--author", help="Filter by author.")] = None,
    source: Annotated[
        str | None, typer.Option("--source", help="Filter by source substring.")
    ] = None,
    storage_dir: Annotated[
        Path, typer.Option("--storage-dir", help="Directory of vector index.")
    ] = Path("./storage/vector_store"),
) -> None:
    """Search indexed documents using hybrid retrieval, reranking, and context compression."""
    from signalrag.embeddings.factory import create_embedding_service
    from signalrag.retrieval import (
        BM25Retriever,
        RetrievalPipeline,
        SemanticRetriever,
    )

    index_path = storage_dir / "index.json"
    if not index_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] No index found at '{index_path}'. Run `signalrag index <path>` first."
        )
        raise typer.Exit(code=1)

    vector_store = MemoryVectorStore(storage_path=index_path)
    vector_store.load(index_path)

    all_chunks = list(vector_store._chunks.values())
    embedding_service = create_embedding_service()

    semantic_retriever = SemanticRetriever(
        vector_store=vector_store, embedding_service=embedding_service
    )
    bm25_retriever = BM25Retriever(chunks=all_chunks)

    pipeline = RetrievalPipeline(
        semantic_retriever=semantic_retriever,
        bm25_retriever=bm25_retriever,
    )

    # Build filters
    filters = {}
    if author:
        filters["author"] = author
    if source:
        filters["source"] = {"$contains": source}

    response = pipeline.retrieve_with_trace(
        query=query,
        top_k=top_k,
        filters=filters if filters else None,
        rerank=rerank,
        compress=compress,
    )
    results = response.results
    trace = response.trace

    if debug:
        trace_table = Table(title="Retrieval Pipeline Diagnostic Trace")
        trace_table.add_column("Stage", style="cyan")
        trace_table.add_column("Details", style="yellow")
        trace_table.add_column("Latency", justify="right", style="green")

        trace_table.add_row("Raw Query", trace.raw_query, "-")
        trace_table.add_row(
            "Rewritten",
            ", ".join(trace.rewritten_queries),
            f"{trace.stages_latency_ms.get('query_rewrite', 0)}ms",
        )
        trace_table.add_row(
            "Hybrid Pool",
            f"{trace.hybrid_candidates_count} candidates",
            f"{trace.stages_latency_ms.get('hybrid_retrieval', 0)}ms",
        )
        trace_table.add_row(
            "Reranker",
            f"{trace.reranked_candidates_count} candidates",
            f"{trace.stages_latency_ms.get('reranking', 0)}ms",
        )
        trace_table.add_row(
            "Compression",
            f"{trace.final_results_count} final",
            f"{trace.stages_latency_ms.get('compression', 0)}ms",
        )
        trace_table.add_row("Total Time", "", f"[bold]{trace.total_latency_ms}ms[/bold]")
        console.print(trace_table)

    if not results:
        console.print("[yellow]No relevant chunks found.[/yellow]")
        return

    table = Table(title=f"Retrieval Results for: '{query}' ({len(results)} matches)")
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Method", style="blue")
    table.add_column("Source", style="magenta")
    table.add_column("Page", justify="center")
    table.add_column("Content Snippet", style="yellow")

    for res in results:
        chunk = res.chunk
        page = str(chunk.metadata.page_number or "-")
        snippet = chunk.text.replace("\n", " ")[:90] + "..."
        table.add_row(
            str(res.rank or 1),
            f"{res.score:.4f}",
            res.retrieval_method,
            Path(chunk.metadata.source).name,
            page,
            snippet,
        )

    console.print(table)


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to ask SignalRAG.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of chunks to retrieve.")] = 5,
    rerank: Annotated[bool, typer.Option("--rerank/--no-rerank", help="Apply reranking.")] = True,
    compress: Annotated[
        bool, typer.Option("--compress/--no-compress", help="Compress retrieved context.")
    ] = False,
    stream: Annotated[
        bool, typer.Option("--stream/--no-stream", help="Stream response tokens.")
    ] = True,
    storage_dir: Annotated[
        Path, typer.Option("--storage-dir", help="Directory of vector index.")
    ] = Path("./storage/vector_store"),
) -> None:
    """Ask a question and receive a grounded answer with inline citations."""
    from rich.markdown import Markdown

    from signalrag.embeddings.factory import create_embedding_service
    from signalrag.generation.engine import SignalRAGEngine
    from signalrag.retrieval import (
        BM25Retriever,
        RetrievalPipeline,
        SemanticRetriever,
    )

    index_path = storage_dir / "index.json"
    if not index_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] No index found at '{index_path}'. Run `signalrag index <path>` first."
        )
        raise typer.Exit(code=1)

    vector_store = MemoryVectorStore(storage_path=index_path)
    vector_store.load(index_path)

    all_chunks = list(vector_store._chunks.values())
    embedding_service = create_embedding_service()

    semantic = SemanticRetriever(vector_store=vector_store, embedding_service=embedding_service)
    bm25 = BM25Retriever(chunks=all_chunks)
    pipeline = RetrievalPipeline(semantic_retriever=semantic, bm25_retriever=bm25)

    engine = SignalRAGEngine(retrieval_pipeline=pipeline)

    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")

    if stream:
        streaming_res = engine.stream_ask(
            question=question,
            top_k=top_k,
            rerank=rerank,
            compress=compress,
        )
        console.print("[bold green]Answer:[/bold green] ", end="")
        citations = []
        for event in streaming_res:
            if event.event_type == "token":
                console.print(event.data, end="", highlight=False)
            elif event.event_type == "citation":
                citations = event.data
        console.print("\n")

        if citations:
            cite_table = Table(title="Sources")
            cite_table.add_column("Ref", justify="center", style="cyan")
            cite_table.add_column("Document Source", style="magenta")
            cite_table.add_column("Page", justify="center")
            cite_table.add_column("Supporting Excerpt", style="yellow")
            for c in citations:
                p = str(c.get("page_number") or "-")
                cite_table.add_row(
                    f"[{c.get('index')}]",
                    Path(c.get("source", "")).name,
                    p,
                    c.get("quote", "")[:90] + "...",
                )
            console.print(cite_table)
    else:
        resp = engine.ask(
            question=question,
            top_k=top_k,
            rerank=rerank,
            compress=compress,
        )
        console.print(Markdown(resp.rendered_markdown))


if __name__ == "__main__":
    app()


@app.command("evaluate")
def evaluate_benchmark(
    dataset_path: Annotated[
        Path, typer.Option("--dataset", "-d", help="Path to evaluation questions dataset")
    ] = Path("eval/questions.json"),
    config_path: Annotated[
        Path, typer.Option("--config", "-c", help="Path to YAML configuration")
    ] = Path("configs/default.yaml"),
    output_path: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional path to write evaluation results JSON"),
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="Optional max questions to evaluate")
    ] = None,
    top_k: Annotated[
        int, typer.Option("--top-k", "-k", help="Number of retrieved chunks per query")
    ] = 5,
) -> None:
    """Run full benchmark evaluation on a test dataset."""
    from signalrag.core.config import load_config
    from signalrag.evaluation.dataset import EvaluationDataset
    from signalrag.evaluation.runner import EvaluationRunner
    from signalrag.generation.engine import RAGEngine

    if not dataset_path.exists():
        console.print(f"[bold red]Dataset not found:[/bold red] {dataset_path}")
        raise typer.Exit(code=1)

    cfg = load_config(config_path)
    engine = RAGEngine.from_config(cfg)
    dataset = EvaluationDataset.from_json(dataset_path)

    console.print(
        f"[bold cyan]Running SignalRAG Evaluation on {len(dataset)} questions...[/bold cyan]"
    )
    runner = EvaluationRunner(engine=engine)
    report = runner.evaluate(dataset=dataset, top_k=top_k, limit=limit)

    console.print("\n[bold green]SignalRAG Evaluation[/bold green]")
    console.print("────────────────────────────────")
    console.print("[bold]Retrieval[/bold]")
    console.print(f"  Recall@5             {report.retrieval_metrics.recall_at_5 * 100:.1f}%")
    console.print(f"  Recall@10            {report.retrieval_metrics.recall_at_10 * 100:.1f}%")
    console.print(f"  MRR                   {report.retrieval_metrics.mrr:.2f}")
    console.print(f"  NDCG@10               {report.retrieval_metrics.ndcg_at_10:.2f}")
    console.print("\n[bold]Generation[/bold]")
    console.print(f"  Faithfulness          {report.avg_faithfulness * 100:.1f}%")
    console.print(f"  Answer Relevance      {report.avg_answer_relevance * 100:.1f}%")
    console.print("\n[bold]Performance[/bold]")
    console.print(f"  P50 latency           {report.latency.p50_s:.1f}s")
    console.print(f"  P95 latency           {report.latency.p95_s:.1f}s")
    console.print(f"  Avg latency           {report.latency.avg_s:.1f}s")
    console.print("────────────────────────────────")
    console.print(f"Evaluation: {report.total_examples} questions\n")

    if output_path:
        report.to_json(output_path)
        console.print(f"[green]Saved evaluation report to {output_path}[/green]")


experiments_app = typer.Typer(help="Manage and inspect evaluation experiment runs.")
app.add_typer(experiments_app, name="experiments")


@experiments_app.command("list")
def list_experiments(
    storage_dir: Annotated[
        Path, typer.Option("--dir", "-d", help="Experiment storage directory")
    ] = Path("storage/experiments"),
) -> None:
    """List tracked evaluation experiments."""
    from signalrag.evaluation.tracker import ExperimentTracker

    tracker = ExperimentTracker(storage_dir=storage_dir)
    runs = tracker.list_runs()

    if not runs:
        console.print("[yellow]No experiments recorded yet in storage/experiments[/yellow]")
        return

    table = Table(title="SignalRAG Experiments", show_lines=True)
    table.add_column("Run ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Recall@5", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Avg Latency", justify="right")

    for r in runs:
        table.add_row(
            r.id,
            r.name,
            f"{r.report.retrieval_metrics.recall_at_5 * 100:.1f}%",
            f"{r.report.retrieval_metrics.mrr:.2f}",
            f"{r.report.avg_faithfulness * 100:.1f}%",
            f"{r.report.latency.avg_s:.2f}s",
        )
    console.print(table)


@experiments_app.command("show")
def show_experiment(
    run_id: Annotated[str, typer.Argument(help="Experiment run ID")],
    storage_dir: Annotated[
        Path, typer.Option("--dir", "-d", help="Experiment storage directory")
    ] = Path("storage/experiments"),
) -> None:
    """Display detailed metrics for an experiment run."""
    from signalrag.evaluation.tracker import ExperimentTracker

    tracker = ExperimentTracker(storage_dir=storage_dir)
    run = tracker.get_run(run_id)
    if not run:
        console.print(f"[red]Experiment {run_id} not found.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Experiment: {run.name} ({run.id})[/bold cyan]")
    console.print(f"Git commit: {run.git_commit}")
    console.print(f"Parameters: {run.config_params}")
    console.print(f"Recall@5: {run.report.retrieval_metrics.recall_at_5 * 100:.1f}%")
    console.print(f"MRR: {run.report.retrieval_metrics.mrr:.2f}")
    console.print(f"Faithfulness: {run.report.avg_faithfulness * 100:.1f}%")


@app.command("dashboard")
def show_dashboard(
    run_id: Annotated[
        str | None,
        typer.Argument(help="Optional experiment run ID (shows latest if omitted)"),
    ] = None,
    storage_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Experiment storage directory"),
    ] = Path("storage/experiments"),
) -> None:
    """Render terminal evaluation dashboard for a benchmark run."""
    from signalrag.evaluation.dashboard import render_evaluation_dashboard, render_failures_table
    from signalrag.evaluation.tracker import ExperimentTracker

    tracker = ExperimentTracker(storage_dir=storage_dir)
    runs = tracker.list_runs()
    if not runs:
        console.print("[yellow]No experiments recorded yet in storage/experiments[/yellow]")
        raise typer.Exit(1)

    target_run = None
    if run_id:
        target_run = tracker.get_run(run_id)
        if not target_run:
            console.print(f"[red]Experiment {run_id} not found.[/red]")
            raise typer.Exit(1)
    else:
        target_run = runs[0]

    render_evaluation_dashboard(target_run.report, console)
    render_failures_table(target_run.report, console)
