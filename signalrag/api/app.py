"""FastAPI application for SignalRAG."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from signalrag.api.schemas import (
    DocumentCreateRequest,
    DocumentResponse,
    EvaluationRunRequest,
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
)
from signalrag.core.config import SignalRAGConfig
from signalrag.evaluation.dataset import EvaluationDataset
from signalrag.evaluation.runner import EvaluationRunner
from signalrag.evaluation.tracker import ExperimentTracker
from signalrag.generation.engine import RAGEngine
from signalrag.models.document import Document

app = FastAPI(
    title="SignalRAG API",
    description="Production-grade RAG pipeline with retrieval and answer evaluation",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global runtime state
_engine: RAGEngine | None = None
_documents: dict[str, Document] = {}
_tracker: ExperimentTracker | None = None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        cfg = SignalRAGConfig()
        cfg.embeddings.provider = "hash"
        cfg.embeddings.dimensions = 64
        _engine = RAGEngine.from_config(cfg)
    return _engine


def get_tracker() -> ExperimentTracker:
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker()
    return _tracker


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "SignalRAG API"}


@app.post("/query", response_model=QueryResponse, tags=["Retrieval & Generation"])
def run_query(request: QueryRequest) -> QueryResponse:
    """Execute end-to-end RAG query."""
    engine = get_engine()
    t0 = time.perf_counter()
    res = engine.query(request.query, top_k=request.top_k)
    lat_ms = (time.perf_counter() - t0) * 1000

    return QueryResponse(
        query=request.query,
        answer=res.answer,
        citations=res.citations,
        latency_ms=round(lat_ms, 2),
        retrieved_chunk_count=len(res.citations),
    )


@app.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
)
def create_document(request: DocumentCreateRequest) -> DocumentResponse:
    """Add a new document to the system."""
    doc_id = request.id or f"doc_{uuid.uuid4().hex[:8]}"
    doc = Document(document_id=doc_id, content=request.content, metadata=request.metadata)
    _documents[doc_id] = doc

    engine = get_engine()
    engine.indexing_pipeline.index_documents([doc])

    return DocumentResponse(
        document_id=doc_id,
        character_count=len(request.content),
        metadata=request.metadata,
    )


@app.get("/documents/{document_id}", response_model=DocumentResponse, tags=["Documents"])
def get_document(document_id: str) -> DocumentResponse:
    """Retrieve document by ID."""
    if document_id not in _documents:
        raise HTTPException(status_code=404, detail="Document not found")
    doc = _documents[document_id]
    return DocumentResponse(
        document_id=doc.document_id,
        character_count=len(doc.content),
        metadata=doc.metadata,
    )


@app.post("/index", response_model=IndexResponse, tags=["Indexing"])
def trigger_index(request: IndexRequest) -> IndexResponse:
    """Trigger indexing of ingested documents."""
    engine = get_engine()
    docs = list(_documents.values())
    engine.indexing_pipeline.index_documents(docs)
    total_chunks = len(engine.indexing_pipeline.vector_store)

    return IndexResponse(
        indexed_documents=len(docs),
        total_chunks=total_chunks,
        status="success",
    )


@app.post("/evaluate", tags=["Evaluation"])
def run_evaluation(request: EvaluationRunRequest) -> dict[str, Any]:
    """Run benchmark evaluation and record experiment."""
    engine = get_engine()
    dataset = EvaluationDataset.from_json(request.dataset_path)
    runner = EvaluationRunner(engine=engine)
    report = runner.evaluate(dataset=dataset, top_k=request.top_k, limit=request.limit)

    tracker = get_tracker()
    run = tracker.log_run(
        name="api_evaluation_run",
        config_params={"top_k": request.top_k, "dataset": request.dataset_path},
        report=report,
    )
    return {"experiment_id": run.id, "report": report.model_dump()}


@app.get("/evaluations", tags=["Evaluation"])
def list_evaluations() -> list[dict[str, Any]]:
    """List all tracked evaluation experiment runs."""
    tracker = get_tracker()
    runs = tracker.list_runs()
    return [r.model_dump() for r in runs]


@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
@app.get("/ui", response_class=HTMLResponse, tags=["Web UI"])
def get_web_ui() -> HTMLResponse:
    """Serve the SignalRAG interactive web UI."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<html><body><h1>SignalRAG</h1><p>Ask a question...</p><div id='sourcesList'>Sources & Citations</div></body></html>"
    )


@app.get("/metrics", tags=["System"])
def get_metrics() -> dict[str, Any]:
    """Export operational latency and token usage metrics."""
    from signalrag.core.observability import global_metrics

    return global_metrics.get_summary()
