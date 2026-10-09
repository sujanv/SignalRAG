"""Tests for IR metrics: Recall@K, Precision@K, MRR, NDCG@K."""

import pytest

from signalrag.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_retrieval_metrics,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    ground_truth = ["doc2", "doc4", "doc5"]

    assert recall_at_k(retrieved, ground_truth, k=1) == 0.0
    assert recall_at_k(retrieved, ground_truth, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, ground_truth, k=4) == pytest.approx(2 / 3)
    assert recall_at_k([], ground_truth, k=5) == 0.0
    assert recall_at_k(retrieved, [], k=5) == 0.0


def test_precision_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    ground_truth = ["doc2", "doc4"]

    assert precision_at_k(retrieved, ground_truth, k=1) == 0.0
    assert precision_at_k(retrieved, ground_truth, k=2) == 0.5
    assert precision_at_k(retrieved, ground_truth, k=4) == 0.5
    assert precision_at_k([], ground_truth, k=2) == 0.0


def test_hit_rate_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    assert hit_rate_at_k(retrieved, ["doc2"], k=1) == 0.0
    assert hit_rate_at_k(retrieved, ["doc2"], k=2) == 1.0
    assert hit_rate_at_k(retrieved, ["doc9"], k=5) == 0.0


def test_mrr():
    assert mrr(["d1", "d2", "d3"], ["d1"]) == 1.0
    assert mrr(["d1", "d2", "d3"], ["d2"]) == 0.5
    assert mrr(["d1", "d2", "d3"], ["d3"]) == pytest.approx(1 / 3)
    assert mrr(["d1", "d2", "d3"], ["d9"]) == 0.0
    assert mrr([], ["d1"]) == 0.0


def test_ndcg_at_k():
    # Perfect ranking
    assert ndcg_at_k(["d1", "d2"], ["d1", "d2"], k=2) == 1.0
    # Hit at first position
    assert ndcg_at_k(["d1", "d9"], ["d1"], k=2) == 1.0
    # Hit at second position vs first
    ndcg_pos2 = ndcg_at_k(["d9", "d1"], ["d1"], k=2)
    assert 0.0 < ndcg_pos2 < 1.0
    assert ndcg_at_k([], ["d1"], k=5) == 0.0


def test_compute_and_aggregate_metrics():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    gt = ["c1", "c3"]
    scores = compute_retrieval_metrics(retrieved, gt)
    assert scores.recall_at_5 == 1.0
    assert scores.precision_at_5 == 0.4
    assert scores.mrr == 1.0
    assert scores.hit_rate_at_5 == 1.0

    # Aggregate
    agg = aggregate_retrieval_metrics([scores, scores])
    assert agg.recall_at_5 == 1.0
    assert agg.mrr == 1.0

    empty_agg = aggregate_retrieval_metrics([])
    assert empty_agg.recall_at_5 == 0.0
