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
    recursive: Annotated[bool, typer.Option("--recursive", "-r", help="Recursively search directory.")] = True,
    page_mode: Annotated[bool, typer.Option("--page-mode", help="Split PDFs into individual page documents.")] = False,
    silent_errors: Annotated[bool, typer.Option("--silent-errors", help="Skip corrupted or unreadable files.")] = False,
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
        snippet = doc.metadata.title or (doc.text[:40].replace("\n", " ") + "..." if doc.text else "")
        table.add_row(doc.id[:10], doc.metadata.file_name, doc.metadata.file_type, pages, chars, snippet)

    console.print(table)
    console.print(f"[bold green]Successfully ingested {len(docs)} structured documents.[/bold green]")


@app.command()
def index(
    path: Annotated[Path, typer.Argument(help="Path to documents to index.")],
    recursive: Annotated[bool, typer.Option("--recursive", "-r", help="Recursively search directory.")] = True,
    force: Annotated[bool, typer.Option("--force", "-f", help="Force reindexing of all documents.")] = False,
    storage_dir: Annotated[Path, typer.Option("--storage-dir", help="Directory to store vector index.")] = Path("./storage/vector_store"),
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
    storage_dir: Annotated[Path, typer.Option("--storage-dir", help="Directory of vector index.")] = Path("./storage/vector_store"),
) -> None:
    """Search indexed documents semantically using vector similarity."""
    index_path = storage_dir / "index.json"
    if not index_path.exists():
        console.print(f"[bold red]Error:[/bold red] No index found at '{index_path}'. Run `signalrag index <path>` first.")
        raise typer.Exit(code=1)

    vector_store = MemoryVectorStore(storage_path=index_path)
    vector_store.load(index_path)

    pipeline = IndexingPipeline(vector_store=vector_store)
    results = pipeline.search(query, top_k=top_k)

    if not results:
        console.print("[yellow]No relevant chunks found.[/yellow]")
        return

    table = Table(title=f"Semantic Search Results for: '{query}' ({len(results)} matches)")
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Score", justify="right", style="green")
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
            Path(chunk.metadata.source).name,
            page,
            snippet,
        )

    console.print(table)


if __name__ == "__main__":
    app()
