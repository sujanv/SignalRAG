"""Tests for BM25 sparse lexical retriever."""

from signalrag.models.chunk import Chunk
from signalrag.retrieval import BM25Retriever


def test_bm25_exact_keyword_matching():
    c1 = Chunk.create(
        "d1", 0, "Error code ERR_AUTH_403 indicates permission denied on access key.", "errors.txt"
    )
    c2 = Chunk.create(
        "d2", 0, "Database connection pool timeout waiting for available connections.", "db.txt"
    )
    c3 = Chunk.create(
        "d3", 0, "Authentication tokens expire after 24 hours of inactivity.", "auth.txt"
    )

    retriever = BM25Retriever(chunks=[c1, c2, c3])

    # Search for rare exact error identifier
    results = retriever.retrieve("ERR_AUTH_403", top_k=2)

    assert len(results) >= 1
    assert results[0].chunk.id == c1.id
    assert results[0].retrieval_method == "bm25"
    assert results[0].rank == 1
    assert 0.0 <= results[0].score <= 1.0


def test_bm25_filtering():
    c1 = Chunk.create("d1", 0, "System architecture overview", "doc.md", category="arch")
    c2 = Chunk.create("d2", 0, "System database indexing overview", "db.md", category="db")

    retriever = BM25Retriever(chunks=[c1, c2])

    results = retriever.retrieve("system overview", top_k=5, filters={"category": "db"})
    assert len(results) == 1
    assert results[0].chunk.id == c2.id


def test_bm25_empty_query_and_empty_corpus():
    empty_retriever = BM25Retriever()
    assert empty_retriever.retrieve("query", top_k=5) == []

    c1 = Chunk.create("d1", 0, "Sample content", "sample.txt")
    retriever = BM25Retriever([c1])
    assert retriever.retrieve("", top_k=5) == []
    assert retriever.retrieve("completely_absent_token_xyz", top_k=5) == []
