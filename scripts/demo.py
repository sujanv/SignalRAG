#!/usr/bin/env python3
"""SignalRAG Interactive End-to-End Showcase Demo."""

from rich.console import Console
from rich.panel import Panel

from signalrag.core.config import SignalRAGConfig
from signalrag.evaluation.dataset import EvaluationDataset
from signalrag.evaluation.runner import EvaluationRunner
from signalrag.generation.engine import RAGEngine
from signalrag.models.document import Document
from signalrag.retrieval.debugger import RetrievalDebugger, render_debug_trace


def main():
    console = Console()
    console.print(
        Panel.fit(
            "[bold green]SignalRAG Portfolio Demonstration[/bold green]\n"
            "[cyan]Production RAG Pipeline & Scientific Evaluation Platform[/cyan]",
            border_style="green",
        )
    )

    # 1. Initialize Engine
    console.print("\n[bold]1. Initializing SignalRAG Engine with Hybrid Pipeline...[/bold]")
    cfg = SignalRAGConfig()
    cfg.embeddings.provider = "hash"
    cfg.embeddings.dimensions = 64
    engine = RAGEngine.from_config(cfg)
    console.print("✓ Configuration loaded. In-memory vector store & BM25 initialized.")

    # 2. Ingest Sample Documents
    console.print("\n[bold]2. Ingesting Corpus Documents...[/bold]")
    docs = [
        Document(
            document_id="doc-bm25-theory",
            content="BM25 is a sparse lexical ranking algorithm that uses term frequency and inverse document frequency with length normalization. BM25Plus avoids negative IDF on small corpora.",
            metadata={"title": "Lexical Search Foundations", "author": "Robertson et al."},
        ),
        Document(
            document_id="doc-hybrid-retrieval",
            content="Hybrid retrieval combines dense vector embeddings with sparse BM25 scores. Reciprocal Rank Fusion (RRF) sums 1/(k + rank) to merge disparate score distributions reliably.",
            metadata={"title": "Hybrid Search at Scale", "author": "Cormack et al."},
        ),
        Document(
            document_id="doc-guardrails",
            content="Grounded-answer guardrails check retrieved context sufficiency before calling the LLM, preventing hallucination when evidence is incomplete.",
            metadata={"title": "Reliable RAG Guardrails", "author": "Lewis et al."},
        ),
    ]
    engine.indexing_pipeline.index_documents(docs)
    console.print(f"✓ Indexed {len(docs)} documents into hybrid index.")

    # 3. Step-by-Step Retrieval Debugger
    query = "How does hybrid retrieval merge dense and sparse rankings?"
    console.print(f"\n[bold]3. Running Retrieval Debugger on query:[/bold] '{query}'")
    debugger = RetrievalDebugger(engine.retrieval_pipeline)
    trace = debugger.trace(query, top_k=3)
    render_debug_trace(trace, console=console)

    # 4. End-to-End Query with Grounded Citations
    console.print("\n[bold]4. Generating Guarded Answer with Verifiable Citations...[/bold]")
    response = engine.query(query, top_k=3)
    console.print(Panel(response.answer, title="[bold green]Synthesized Answer[/bold green]"))

    console.print("[bold]Extracted Citations:[/bold]")
    for idx, c in enumerate(response.citations, start=1):
        console.print(f"  [{idx}] Document: [cyan]{c.document_id}[/cyan] (Chunk: {c.chunk_id})")
        console.print(f"      [dim]{c.snippet}[/dim]")

    # 5. Run Evaluation Benchmark
    console.print("\n[bold]5. Running Evaluation Benchmark...[/bold]")
    dataset = EvaluationDataset.from_json("eval/questions.json")
    runner = EvaluationRunner(engine=engine)
    report = runner.evaluate(dataset=dataset, top_k=3, limit=5)
    console.print(f"✓ Benchmark finished on {report.total_examples} questions.")
    console.print(f"  Recall@5: [green]{report.retrieval_metrics.recall_at_5 * 100:.1f}%[/green]")
    console.print(f"  MRR:      [green]{report.retrieval_metrics.mrr:.2f}[/green]")
    console.print(f"  P50 Latency: [yellow]{report.latency.p50_s:.2f}s[/yellow]")

    console.print("\n[bold green]✓ Demo completed successfully![/bold green]\n")


if __name__ == "__main__":
    main()
