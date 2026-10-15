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


@app.command("debug")
def debug_retrieval(
    query_text: Annotated[str, typer.Argument(help="Query to run through retrieval debugger")],
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to YAML configuration"),
    ] = Path("configs/default.yaml"),
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Top-K candidate items to return"),
    ] = 5,
) -> None:
    """Inspect step-by-step retrieval stages for a query."""
    from signalrag.core.config import load_config
    from signalrag.generation.engine import RAGEngine
    from signalrag.retrieval.debugger import RetrievalDebugger, render_debug_trace

    cfg = load_config(config_path)
    engine = RAGEngine.from_config(cfg)
    debugger = RetrievalDebugger(engine.retrieval_pipeline)
    trace = debugger.trace(query_text, top_k=top_k)
    render_debug_trace(trace, console=console)


@app.command("analyze-failures")
def analyze_failures(
    run_id: Annotated[
        str | None,
        typer.Argument(help="Experiment run ID (analyzes latest if omitted)"),
    ] = None,
    storage_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Experiment storage directory"),
    ] = Path("storage/experiments"),
) -> None:
    """Categorize and diagnose failure modes for an evaluation run."""
    from signalrag.evaluation.failure_analysis import FailureAnalyzer, render_failure_analysis
    from signalrag.evaluation.tracker import ExperimentTracker

    tracker = ExperimentTracker(storage_dir=storage_dir)
    runs = tracker.list_runs()
    if not runs:
        console.print("[yellow]No experiments found in storage/experiments[/yellow]")
        raise typer.Exit(1)

    target_run = tracker.get_run(run_id) if run_id else runs[0]
    if not target_run:
        console.print(f"[red]Experiment {run_id} not found[/red]")
        raise typer.Exit(1)

    analyzer = FailureAnalyzer()
    summary = analyzer.analyze(target_run.report)
    render_failure_analysis(summary, console=console)


@app.command("compare")
def compare_runs(
    run_a: Annotated[str, typer.Argument(help="First experiment run ID")],
    run_b: Annotated[str, typer.Argument(help="Second experiment run ID")],
    storage_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Experiment storage directory"),
    ] = Path("storage/experiments"),
) -> None:
    """Compare two evaluation runs side by side."""
    from signalrag.evaluation.comparison import compare_experiments, render_comparison_table
    from signalrag.evaluation.tracker import ExperimentTracker

    tracker = ExperimentTracker(storage_dir=storage_dir)
    exp_a = tracker.get_run(run_a)
    exp_b = tracker.get_run(run_b)

    if not exp_a:
        console.print(f"[red]Experiment {run_a} not found[/red]")
        raise typer.Exit(1)
    if not exp_b:
        console.print(f"[red]Experiment {run_b} not found[/red]")
        raise typer.Exit(1)

    comp = compare_experiments(exp_a, exp_b)
    render_comparison_table(comp, console=console)


@app.command("serve")
def serve_api(
    host: Annotated[str, typer.Option("--host", "-h", help="Bind host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Bind port")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Enable auto-reload")] = False,
) -> None:
    """Launch SignalRAG FastAPI server with OpenAPI documentation."""
    import uvicorn

    console.print(f"[bold green]Starting SignalRAG API at http://{host}:{port}[/bold green]")
    console.print(f"OpenAPI Documentation: http://{host}:{port}/docs")
    uvicorn.run("signalrag.api.app:app", host=host, port=port, reload=reload)


@app.command("multihop")
def multihop_query(
    query_text: Annotated[str, typer.Argument(help="Complex query to decompose and retrieve")],
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to YAML configuration"),
    ] = Path("configs/default.yaml"),
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Candidate items per sub-query"),
    ] = 3,
) -> None:
    """Decompose complex queries into sub-questions and retrieve multi-hop evidence."""
    from signalrag.core.config import load_config
    from signalrag.generation.engine import RAGEngine
    from signalrag.retrieval.decomposer import MultiHopRetriever

    cfg = load_config(config_path)
    engine = RAGEngine.from_config(cfg)
    retriever = MultiHopRetriever(pipeline=engine.retrieval_pipeline)

    console.print(f"[bold cyan]Original Query:[/bold cyan] {query_text}")
    trace = retriever.retrieve(query_text, top_k_per_subquery=top_k)

    console.print(
        f"[bold green]Decomposition Plan:[/bold green] ({len(trace.plan.sub_queries)} sub-queries)"
    )
    for sq in trace.plan.sub_queries:
        console.print(f"  • [yellow]{sq.id}[/yellow]: {sq.query} [dim]({sq.intent})[/dim]")

    console.print(
        f"\n[bold green]Merged Evidence Chunks:[/bold green] ({len(trace.merged_results)} unique passages)"
    )
    for idx, r in enumerate(trace.merged_results, 1):
        console.print(
            f"  [{idx}] Chunk {r.chunk.id} (Doc: {r.chunk.document_id}) - Score: {r.score:.3f}"
        )
        console.print(f"      [dim]{r.chunk.text[:120]}...[/dim]")


@app.command("generate-eval")
def generate_eval(
    docs_dir: Annotated[
        Path,
        typer.Option("--docs", "-d", help="Directory of documents to ingest and synthesize from"),
    ] = Path("data/samples"),
    output_path: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path for evaluation JSON dataset"),
    ] = Path("eval/synthetic_eval.json"),
    max_count: Annotated[
        int,
        typer.Option("--count", "-n", help="Maximum evaluation pairs to generate"),
    ] = 10,
) -> None:
    """Generate synthetic evaluation question-answer dataset from local corpus."""
    from signalrag.chunking.recursive import RecursiveCharacterChunker
    from signalrag.evaluation.synthetic import SyntheticDatasetGenerator
    from signalrag.ingestion.directory_loader import DirectoryLoader

    loader = DirectoryLoader(docs_dir)
    docs = loader.load()
    if not docs:
        console.print(
            f"[yellow]No documents found in {docs_dir}. Generating fallback sample dataset...[/yellow]"
        )
        from signalrag.models.document import Document

        docs = [
            Document(
                id="doc_demo",
                text="SignalRAG provides hybrid retrieval combining dense vector embeddings and BM25 lexical ranking.",
                metadata={"source": "demo.txt"},
            )
        ]

    chunker = RecursiveCharacterChunker(chunk_size=300, chunk_overlap=30)
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunker.chunk(d))

    generator = SyntheticDatasetGenerator()
    dataset = generator.generate_from_chunks(all_chunks, max_examples=max_count)
    dataset.to_json(output_path)
    console.print(
        f"[bold green]Successfully generated {len(dataset)} synthetic evaluation examples into {output_path}![/bold green]"
    )


@app.command("route")
def route_query(
    query_text: Annotated[str, typer.Argument(help="Query text to analyze for semantic routing")],
) -> None:
    """Analyze query and determine optimal retrieval strategy (BM25, Semantic, MultiHop, or Hybrid)."""
    from signalrag.retrieval.router import QueryRouter

    router = QueryRouter()
    decision = router.route(query_text)

    console.print(f"[bold cyan]Query:[/bold cyan] {decision.query}")
    console.print(
        f"[bold green]Routed Strategy:[/bold green] [yellow]{decision.target_strategy.upper()}[/yellow]"
    )
    console.print(f"[bold]Confidence:[/bold] {decision.confidence * 100:.1f}%")
    console.print(f"[dim]Reasoning: {decision.reasoning}[/dim]")


@app.command("verify-answer")
def verify_answer_cli(
    answer_text: Annotated[
        str, typer.Option("--answer", "-a", help="Generated answer text to audit")
    ],
    context_text: Annotated[str, typer.Option("--context", "-c", help="Retrieved context passage")],
) -> None:
    """Verify factual grounding of answer against evidence to flag hallucinations."""
    from signalrag.generation.hallucination import HallucinationDetector
    from signalrag.models.chunk import Chunk

    chunk = Chunk(id="ref_passage", document_id="doc1", text=context_text)
    detector = HallucinationDetector(support_threshold=0.5)
    report = detector.verify_answer(answer_text, [chunk])

    status_color = (
        "green"
        if report.verdict == "PASS"
        else ("yellow" if report.verdict == "WARNING" else "red")
    )
    console.print(
        f"[bold]Attribution Verdict:[/bold] [{status_color}]{report.verdict}[/{status_color}] ({report.faithfulness_ratio * 100:.1f}% grounded)"
    )
    console.print(
        f"Total Claims: {report.total_claims} | Supported: {report.supported_claims} | Hallucinations: {report.unsupported_claims}\n"
    )

    for idx, c in enumerate(report.claims, 1):
        mark = "✓" if c.supported else "✗"
        color = "green" if c.supported else "red"
        console.print(
            f"  [{color}]{mark}[/{color}] Claim {idx}: {c.claim} [dim](score: {c.confidence:.2f})[/dim]"
        )


@app.command("expand")
def expand_query_cli(
    query_text: Annotated[str, typer.Argument(help="Initial query to expand using PRF")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Candidate items to retrieve")] = 3,
) -> None:
    """Enrich query with pseudo-relevance feedback terms from top retrieval passages."""
    from signalrag.core.config import load_config
    from signalrag.generation.engine import RAGEngine
    from signalrag.retrieval.expansion import PseudoRelevanceExpander

    cfg = load_config()
    engine = RAGEngine.from_config(cfg)
    expander = PseudoRelevanceExpander(top_n_terms=3)

    exp_res, results = expander.retrieve_with_prf(
        query_text, engine.retrieval_pipeline, top_k=top_k
    )
    console.print(f"[bold cyan]Original Query:[/bold cyan] {exp_res.original_query}")
    console.print(
        f"[bold green]Discovered Terms:[/bold green] {', '.join(exp_res.expansion_terms) if exp_res.expansion_terms else 'None'}"
    )
    console.print(f"[bold yellow]Expanded Query:[/bold yellow] {exp_res.expanded_query}")
    console.print(f"\n[bold]Retrieved Chunks ({len(results)}):[/bold]")
    for idx, r in enumerate(results, 1):
        console.print(
            f"  [{idx}] Chunk {r.chunk.id} (Score: {r.score:.3f}): {r.chunk.text[:100]}..."
        )


@app.command("versions")
def show_versions_cli(
    doc_id: Annotated[str, typer.Argument(help="Document identifier to inspect revision history")],
) -> None:
    """Show document version history and superseded revisions."""
    from signalrag.indexing.versioning import DocumentVersionManager

    # For demonstration/CLI inspection
    mgr = DocumentVersionManager()
    history = mgr.get_version_history(doc_id)
    if not history:
        console.print(f"[yellow]No version history found for document '{doc_id}'.[/yellow]")
        return

    console.print(f"[bold cyan]Document Revisions for:[/bold cyan] {doc_id}")
    for rev in history:
        status_tag = "[green](Active)[/green]" if rev.is_active else "[red](Superseded)[/red]"
        console.print(f"  • Version {rev.version} {status_tag} - Chunks: {len(rev.chunk_ids)}")


@app.command("budget")
def budget_cli(
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", "-m", help="Total context budget in tokens")
    ] = 1024,
    query_text: Annotated[str, typer.Option("--query", "-q", help="Query text")] = "architecture",
) -> None:
    """Inspect dynamic token budget allocation for retrieved chunks."""
    from signalrag.generation.budget import TokenBudgetAllocator
    from signalrag.models.chunk import Chunk, ChunkMetadata

    chunks = [
        Chunk(
            id=f"chunk_{i}",
            document_id=f"doc_{i}",
            text=f"Passage {i} detailing system specifications and parameters. " * 15,
            metadata=ChunkMetadata(document_id=f"doc_{i}", chunk_index=i, source="spec.txt"),
        )
        for i in range(5)
    ]

    allocator = TokenBudgetAllocator(max_context_tokens=max_tokens)
    res = allocator.allocate(chunks, query=query_text)

    console.print(
        f"[bold cyan]Token Budget Allocator[/bold cyan] (Limit: {res.budget_limit} tokens)"
    )
    console.print(f"Tokens Used: [green]{res.total_tokens_used}[/green] / {res.budget_limit}")
    console.print(
        f"Allocated Chunks: {len(res.allocated_chunks)} | Dropped Chunks: {res.dropped_chunk_count}"
    )


@app.command("dedup")
def dedup_cli(
    threshold: Annotated[
        float, typer.Option("--threshold", "-t", help="Jaccard similarity threshold")
    ] = 0.75,
) -> None:
    """Analyze chunk redundancy and demonstrate near-duplicate clustering."""
    from signalrag.chunking.dedup import SemanticChunkDeduplicator
    from signalrag.models.chunk import Chunk, ChunkMetadata

    chunks = [
        Chunk(
            id="c1",
            document_id="doc1",
            text="Vector embeddings capture semantic similarity across high-dimensional space.",
            metadata=ChunkMetadata(document_id="doc1", chunk_index=0, source="s1"),
        ),
        Chunk(
            id="c2",
            document_id="doc2",
            text="Vector embeddings capture semantic similarity in high-dimensional space.",
            metadata=ChunkMetadata(document_id="doc2", chunk_index=1, source="s2"),
        ),
        Chunk(
            id="c3",
            document_id="doc3",
            text="BM25 is a ranking function based on probabilistic relevance models.",
            metadata=ChunkMetadata(document_id="doc3", chunk_index=2, source="s3"),
        ),
    ]

    dedup = SemanticChunkDeduplicator(similarity_threshold=threshold)
    unique_chunks, report = dedup.deduplicate(chunks)

    console.print(f"[bold cyan]Chunk Deduplication[/bold cyan] (Threshold: {threshold:.2f})")
    console.print(
        f"Original: {report.original_count} | Retained: [green]{report.retained_count}[/green] | Pruned: [red]{report.duplicates_removed}[/red]"
    )


@app.command("auth-query")
def auth_query_cli(
    user_id: Annotated[str, typer.Option("--user", "-u", help="User identifier")] = "alice",
    tenant_id: Annotated[
        str, typer.Option("--tenant", "-t", help="Tenant partition")
    ] = "org_engineering",
    role: Annotated[
        str, typer.Option("--role", "-r", help="Assigned role (reader, contributor, admin)")
    ] = "reader",
) -> None:
    """Simulate RBAC-scoped retrieval under multi-tenant security context."""
    from signalrag.core.security import AccessController, SecurityContext
    from signalrag.models.chunk import Chunk, ChunkMetadata

    chunks = [
        Chunk(
            id="c_eng",
            document_id="d1",
            text="Engineering internal architecture docs.",
            metadata=ChunkMetadata(
                document_id="d1",
                chunk_index=0,
                source="eng.txt",
                extra={"tenant_id": "org_engineering", "min_role": "reader"},
            ),
        ),
        Chunk(
            id="c_hr",
            document_id="d2",
            text="HR confidential compensation guidelines.",
            metadata=ChunkMetadata(
                document_id="d2",
                chunk_index=0,
                source="hr.txt",
                extra={"tenant_id": "org_hr", "min_role": "admin"},
            ),
        ),
    ]

    sec_ctx = SecurityContext(user_id=user_id, tenant_id=tenant_id, roles=[role])
    ctrl = AccessController()
    visible = ctrl.filter_chunks(sec_ctx, chunks)

    console.print(
        f"[bold cyan]RBAC Context:[/bold cyan] User: {user_id} | Tenant: {tenant_id} | Role: {role}"
    )
    console.print(f"Visible Chunks: [green]{len(visible)}[/green] / {len(chunks)}")
    for c in visible:
        console.print(f"  • {c.id}: {c.text}")


@app.command("regression")
def check_regression_cli(
    max_drop: Annotated[
        float, typer.Option("--max-drop", "-d", help="Max allowable drop in metric value")
    ] = 0.05,
) -> None:
    """Run regression check comparing simulated candidate metrics against baseline."""
    from signalrag.evaluation.regression import RegressionGuard

    base = {"recall_at_5": 0.88, "mrr": 0.82, "faithfulness": 0.92}
    cand = {"recall_at_5": 0.89, "mrr": 0.81, "faithfulness": 0.91}

    guard = RegressionGuard(max_allowed_drop=max_drop)
    res = guard.check(base, cand)

    status_tag = "[bold green]PASSED[/bold green]" if res.passed else "[bold red]FAILED[/bold red]"
    console.print(f"Regression Check: {status_tag} (Max Allowed Drop: {max_drop * 100:.1f}%)")
    for d in res.deltas:
        mark = "✓" if not d.is_regression else "✗"
        console.print(
            f"  {mark} {d.metric_name}: Base={d.baseline_value:.3f} | Cand={d.candidate_value:.3f} | Delta={d.diff:+.3f}"
        )


@app.command("graph-query")
def graph_query_cli(
    query_text: Annotated[
        str, typer.Argument(help="Query to evaluate against entity knowledge graph")
    ],
) -> None:
    """Traverse associative entity-relation graph and retrieve linked passage subgraphs."""
    from signalrag.models.chunk import Chunk, ChunkMetadata
    from signalrag.retrieval.graph import KnowledgeGraphRetriever

    # Demo corpus chunks
    chunks = [
        Chunk(
            id="c_graph_1",
            document_id="doc1",
            text="SignalRAG utilizes BM25 for lexical search. BM25 implements probabilistic ranking.",
            metadata=ChunkMetadata(document_id="doc1", chunk_index=0, source="s1"),
        ),
        Chunk(
            id="c_graph_2",
            document_id="doc2",
            text="SignalRAG connects to SQLite for persistent vector storage.",
            metadata=ChunkMetadata(document_id="doc2", chunk_index=1, source="s2"),
        ),
    ]

    kg = KnowledgeGraphRetriever()
    kg.index_chunks(chunks)
    ctx = kg.retrieve_subgraph(query_text)

    console.print(f"[bold cyan]GraphRAG Query:[/bold cyan] {query_text}")
    console.print(
        f"Entities Found: [green]{len(ctx.entities)}[/green] | Relations: [green]{len(ctx.relations)}[/green]"
    )
    if ctx.subgraph_text:
        console.print(f"[bold yellow]Subgraphs Discovered:[/bold yellow]\n{ctx.subgraph_text}")
    console.print(f"Connected Chunk IDs: {ctx.connected_chunk_ids}")
