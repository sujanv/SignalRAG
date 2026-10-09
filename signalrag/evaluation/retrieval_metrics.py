"""Information retrieval evaluation metrics: Recall@K, Precision@K, MRR, NDCG@K."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, Field


def recall_at_k(retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str], k: int) -> float:
    """Calculate Recall@K: proportion of relevant items retrieved in top-K."""
    if not ground_truth_ids:
        return 0.0
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    gt_set = set(ground_truth_ids)
    hits = sum(1 for item in top_k if item in gt_set)
    return hits / len(gt_set)


def precision_at_k(retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str], k: int) -> float:
    """Calculate Precision@K: proportion of top-K retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    gt_set = set(ground_truth_ids)
    hits = sum(1 for item in top_k if item in gt_set)
    return hits / len(top_k)


def hit_rate_at_k(retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str], k: int) -> float:
    """Calculate HitRate@K (1.0 if at least one relevant item in top-K, else 0.0)."""
    if not ground_truth_ids or k <= 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    gt_set = set(ground_truth_ids)
    return 1.0 if (top_k & gt_set) else 0.0


def mrr(retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) for a single query.

    Returns 1 / rank of the first relevant document, or 0.0 if none retrieved.
    """
    if not ground_truth_ids:
        return 0.0
    gt_set = set(ground_truth_ids)
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in gt_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K (binary relevance)."""
    if not ground_truth_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    gt_set = set(ground_truth_ids)

    # DCG calculation
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in gt_set:
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG calculation
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


class RetrievalMetricScores(BaseModel):
    """Container for calculated retrieval metrics for a query or corpus."""

    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    hit_rate_at_5: float = 0.0
    hit_rate_at_10: float = 0.0
    details: dict[str, float] = Field(default_factory=dict)


def compute_retrieval_metrics(
    retrieved_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> RetrievalMetricScores:
    """Compute all standard retrieval metrics for a single search result list."""
    scores = RetrievalMetricScores(
        recall_at_1=recall_at_k(retrieved_ids, ground_truth_ids, 1),
        recall_at_3=recall_at_k(retrieved_ids, ground_truth_ids, 3),
        recall_at_5=recall_at_k(retrieved_ids, ground_truth_ids, 5),
        recall_at_10=recall_at_k(retrieved_ids, ground_truth_ids, 10),
        precision_at_1=precision_at_k(retrieved_ids, ground_truth_ids, 1),
        precision_at_3=precision_at_k(retrieved_ids, ground_truth_ids, 3),
        precision_at_5=precision_at_k(retrieved_ids, ground_truth_ids, 5),
        precision_at_10=precision_at_k(retrieved_ids, ground_truth_ids, 10),
        mrr=mrr(retrieved_ids, ground_truth_ids),
        ndcg_at_5=ndcg_at_k(retrieved_ids, ground_truth_ids, 5),
        ndcg_at_10=ndcg_at_k(retrieved_ids, ground_truth_ids, 10),
        hit_rate_at_5=hit_rate_at_k(retrieved_ids, ground_truth_ids, 5),
        hit_rate_at_10=hit_rate_at_k(retrieved_ids, ground_truth_ids, 10),
    )
    for k in ks:
        scores.details[f"recall@{k}"] = recall_at_k(retrieved_ids, ground_truth_ids, k)
        scores.details[f"precision@{k}"] = precision_at_k(retrieved_ids, ground_truth_ids, k)
        scores.details[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, ground_truth_ids, k)
    return scores


def aggregate_retrieval_metrics(
    scores_list: Sequence[RetrievalMetricScores],
) -> RetrievalMetricScores:
    """Aggregate a sequence of metric scores into mean values across queries."""
    if not scores_list:
        return RetrievalMetricScores()

    n = len(scores_list)
    avg_rec1 = sum(s.recall_at_1 for s in scores_list) / n
    avg_rec3 = sum(s.recall_at_3 for s in scores_list) / n
    avg_rec5 = sum(s.recall_at_5 for s in scores_list) / n
    avg_rec10 = sum(s.recall_at_10 for s in scores_list) / n
    avg_prec1 = sum(s.precision_at_1 for s in scores_list) / n
    avg_prec3 = sum(s.precision_at_3 for s in scores_list) / n
    avg_prec5 = sum(s.precision_at_5 for s in scores_list) / n
    avg_prec10 = sum(s.precision_at_10 for s in scores_list) / n
    avg_mrr = sum(s.mrr for s in scores_list) / n
    avg_ndcg5 = sum(s.ndcg_at_5 for s in scores_list) / n
    avg_ndcg10 = sum(s.ndcg_at_10 for s in scores_list) / n
    avg_hit5 = sum(s.hit_rate_at_5 for s in scores_list) / n
    avg_hit10 = sum(s.hit_rate_at_10 for s in scores_list) / n

    return RetrievalMetricScores(
        recall_at_1=round(avg_rec1, 4),
        recall_at_3=round(avg_rec3, 4),
        recall_at_5=round(avg_rec5, 4),
        recall_at_10=round(avg_rec10, 4),
        precision_at_1=round(avg_prec1, 4),
        precision_at_3=round(avg_prec3, 4),
        precision_at_5=round(avg_prec5, 4),
        precision_at_10=round(avg_prec10, 4),
        mrr=round(avg_mrr, 4),
        ndcg_at_5=round(avg_ndcg5, 4),
        ndcg_at_10=round(avg_ndcg10, 4),
        hit_rate_at_5=round(avg_hit5, 4),
        hit_rate_at_10=round(avg_hit10, 4),
    )
