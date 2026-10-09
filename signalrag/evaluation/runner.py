"""Automated evaluation runner for benchmarking retrieval and answer generation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from signalrag.evaluation.answer_metrics import AnswerEvaluationResult, evaluate_answer
from signalrag.evaluation.dataset import EvaluationDataset
from signalrag.evaluation.retrieval_metrics import (
    RetrievalMetricScores,
    aggregate_retrieval_metrics,
    compute_retrieval_metrics,
)
from signalrag.generation.engine import RAGEngine
from signalrag.retrieval.pipeline import RetrievalPipeline


class ExampleResult(BaseModel):
    """Detailed outcome for an individual evaluation example."""

    example_id: str
    question: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    generated_answer: str = ""
    retrieval_scores: RetrievalMetricScores
    generation_scores: AnswerEvaluationResult | None = None
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    success: bool = True
    error_message: str | None = None


class LatencyStats(BaseModel):
    """Latency distribution summary in milliseconds and seconds."""

    count: int = 0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    avg_ms: float = 0.0
    p50_s: float = 0.0
    p95_s: float = 0.0
    avg_s: float = 0.0


class EvaluationReport(BaseModel):
    """Consolidated benchmark report containing metrics, latencies, and example records."""

    benchmark_name: str = "SignalRAG Evaluation"
    timestamp: float = Field(default_factory=time.time)
    total_examples: int = 0
    successful_examples: int = 0
    retrieval_metrics: RetrievalMetricScores
    avg_faithfulness: float = 0.0
    avg_answer_relevance: float = 0.0
    avg_citation_precision: float = 0.0
    latency: LatencyStats
    results: list[ExampleResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        """Write evaluation report to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(self.model_dump_json(indent=indent), encoding="utf-8")


class EvaluationRunner:
    """Executes evaluation benchmarks over datasets using retrieval and generation components."""

    def __init__(
        self,
        engine: RAGEngine | None = None,
        retrieval_pipeline: RetrievalPipeline | None = None,
    ):
        self.engine = engine
        self.pipeline = retrieval_pipeline or (engine.retrieval_pipeline if engine else None)

    def evaluate(
        self,
        dataset: EvaluationDataset,
        top_k: int = 5,
        evaluate_generation: bool = True,
        limit: int | None = None,
    ) -> EvaluationReport:
        """Run benchmark over dataset and produce consolidated report."""
        examples = dataset.examples[:limit] if limit else dataset.examples
        results: list[ExampleResult] = []
        retrieval_scores_list: list[RetrievalMetricScores] = []
        faithfulness_scores: list[float] = []
        relevance_scores: list[float] = []
        citation_scores: list[float] = []
        latencies_ms: list[float] = []

        for ex in examples:
            t0 = time.perf_counter()
            retrieval_time_ms = 0.0
            gen_time_ms = 0.0

            try:
                # 1. Retrieval
                t_ret_start = time.perf_counter()
                if self.pipeline:
                    retrieval_results = self.pipeline.retrieve(ex.question, top_k=top_k)
                else:
                    retrieval_results = []
                retrieval_time_ms = (time.perf_counter() - t_ret_start) * 1000

                retrieved_chunks = [r.chunk for r in retrieval_results]
                retrieved_chunk_ids = [c.chunk_id for c in retrieved_chunks]
                retrieved_doc_ids = list(dict.fromkeys(c.document_id for c in retrieved_chunks))

                target_ids = ex.expected_chunk_ids if ex.expected_chunk_ids else ex.expected_doc_ids
                candidate_ids = retrieved_chunk_ids if ex.expected_chunk_ids else retrieved_doc_ids

                r_scores = compute_retrieval_metrics(candidate_ids, target_ids)
                retrieval_scores_list.append(r_scores)

                # 2. Generation & Answer Evaluation
                gen_scores = None
                generated_answer = ""
                if evaluate_generation and self.engine:
                    t_gen_start = time.perf_counter()
                    rag_res = self.engine.query(ex.question, top_k=top_k)
                    gen_time_ms = (time.perf_counter() - t_gen_start) * 1000
                    generated_answer = rag_res.answer

                    context_texts = [c.content for c in retrieved_chunks]
                    gen_scores = evaluate_answer(
                        question=ex.question,
                        generated_answer=generated_answer,
                        context_chunks=context_texts,
                        ground_truth=ex.ground_truth_answer,
                        citations=rag_res.citations,
                    )
                    faithfulness_scores.append(gen_scores.faithfulness)
                    relevance_scores.append(gen_scores.answer_relevance)
                    citation_scores.append(gen_scores.citation_precision)

                total_time_ms = (time.perf_counter() - t0) * 1000
                latencies_ms.append(total_time_ms)

                results.append(
                    ExampleResult(
                        example_id=ex.id,
                        question=ex.question,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        retrieved_doc_ids=retrieved_doc_ids,
                        generated_answer=generated_answer,
                        retrieval_scores=r_scores,
                        generation_scores=gen_scores,
                        retrieval_latency_ms=round(retrieval_time_ms, 2),
                        generation_latency_ms=round(gen_time_ms, 2),
                        total_latency_ms=round(total_time_ms, 2),
                        success=True,
                    )
                )

            except Exception as e:
                total_time_ms = (time.perf_counter() - t0) * 1000
                latencies_ms.append(total_time_ms)
                r_scores = compute_retrieval_metrics(
                    [], ex.expected_chunk_ids or ex.expected_doc_ids
                )
                retrieval_scores_list.append(r_scores)
                results.append(
                    ExampleResult(
                        example_id=ex.id,
                        question=ex.question,
                        retrieval_scores=r_scores,
                        total_latency_ms=round(total_time_ms, 2),
                        success=False,
                        error_message=str(e),
                    )
                )

        # Aggregate Metrics
        aggregated_retrieval = aggregate_retrieval_metrics(retrieval_scores_list)
        avg_faith = (
            round(sum(faithfulness_scores) / len(faithfulness_scores), 4)
            if faithfulness_scores
            else 0.0
        )
        avg_rel = (
            round(sum(relevance_scores) / len(relevance_scores), 4) if relevance_scores else 0.0
        )
        avg_cit = round(sum(citation_scores) / len(citation_scores), 4) if citation_scores else 0.0

        lat_arr = np.array(latencies_ms) if latencies_ms else np.array([0.0])
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p95 = float(np.percentile(lat_arr, 95))
        avg_lat = float(np.mean(lat_arr))

        latency_summary = LatencyStats(
            count=len(latencies_ms),
            p50_ms=round(p50, 2),
            p90_ms=round(p90, 2),
            p95_ms=round(p95, 2),
            avg_ms=round(avg_lat, 2),
            p50_s=round(p50 / 1000.0, 2),
            p95_s=round(p95 / 1000.0, 2),
            avg_s=round(avg_lat / 1000.0, 2),
        )

        return EvaluationReport(
            benchmark_name=dataset.name,
            total_examples=len(examples),
            successful_examples=sum(1 for r in results if r.success),
            retrieval_metrics=aggregated_retrieval,
            avg_faithfulness=avg_faith,
            avg_answer_relevance=avg_rel,
            avg_citation_precision=avg_cit,
            latency=latency_summary,
            results=results,
        )
