"""Tests for second-stage rerankers."""

from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult
from signalrag.retrieval import CrossEncoderReranker, HeuristicCrossReranker


def test_heuristic_reranker_promotes_exact_matches():
    # c1 is a partial match that scored high in first stage
    c1 = Chunk.create(
        "d1", 0, "General information about algorithms and computer science principles.", "cs.txt"
    )
    # c2 contains exact phrase match and section title match
    c2 = Chunk.create(
        "d2",
        0,
        "The cross-encoder architecture computes joint attention over query and candidate passage pairs.",
        "rerank.txt",
        section_title="Cross-Encoder Architecture",
    )

    r1 = SearchResult(chunk=c1, score=0.85, rank=1, retrieval_method="vector")
    r2 = SearchResult(chunk=c2, score=0.70, rank=2, retrieval_method="vector")

    reranker = HeuristicCrossReranker()
    reranked = reranker.rerank("cross-encoder architecture", [r1, r2], top_k=2)

    assert len(reranked) == 2
    # c2 should now be promoted to rank 1
    assert reranked[0].chunk.id == c2.id
    assert reranked[0].rank == 1
    assert reranked[0].retrieval_method == "reranked"
    assert reranked[0].rerank_score is not None
    assert reranked[0].rerank_score > reranked[1].rerank_score


def test_cross_encoder_reranker_fallback():
    c1 = Chunk.create("d1", 0, "Database indexing using B-Trees and LSM Trees.", "db.txt")
    r1 = SearchResult(chunk=c1, score=0.6, rank=1)

    reranker = CrossEncoderReranker(model_name="non_existent_mock_model")
    reranked = reranker.rerank("database indexing", [r1], top_k=1)

    assert len(reranked) == 1
    assert reranked[0].rank == 1
    assert reranked[0].retrieval_method == "reranked"


def test_reranker_empty_inputs():
    reranker = HeuristicCrossReranker()
    assert reranker.rerank("", [], top_k=3) == []
    assert reranker.rerank("query", [], top_k=3) == []
