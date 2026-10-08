"""Main entry point for SignalRAG CLI."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from signalrag import __version__
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


if __name__ == "__main__":
    app()
