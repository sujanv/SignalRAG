"""Evaluation suite for SignalRAG retrieval and generation."""

from signalrag.evaluation.dataset import EvalExample, EvaluationDataset
from signalrag.evaluation.retrieval_metrics import (
    RetrievalMetricScores,
    aggregate_retrieval_metrics,
    compute_retrieval_metrics,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "EvalExample",
    "EvaluationDataset",
    "RetrievalMetricScores",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "hit_rate_at_k",
    "compute_retrieval_metrics",
    "aggregate_retrieval_metrics",
]
