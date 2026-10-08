"""Main entry point for SignalRAG CLI."""

import typer
from rich.console import Console

from signalrag import __version__

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


if __name__ == "__main__":
    app()
