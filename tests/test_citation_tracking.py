"""Tests for citation tracking and claim provenance."""

from signalrag.generation import CitationTracker
from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult


def test_citation_tracker_extracts_indices():
    tracker = CitationTracker()
    text = "SignalRAG supports BM25 [1] and reciprocal rank fusion [2, 3]. It also features reranking [1]."
    indices = tracker.extract_citation_indices(text)
    assert indices == [1, 2, 3]


def test_citation_tracker_claims_mapping():
    tracker = CitationTracker()

    c1 = Chunk.create(
        "d1",
        0,
        "Hybrid search merges BM25 and vector scores with alpha interpolation.",
        "search.md",
        page_number=4,
    )
    c2 = Chunk.create(
        "d2",
        0,
        "Cross-encoders score candidate chunks based on joint attention.",
        "rerank.md",
        page_number=10,
    )

    r1 = SearchResult(chunk=c1, score=0.92, rank=1)
    r2 = SearchResult(chunk=c2, score=0.88, rank=2)

    answer_text = (
        "SignalRAG performs hybrid search using score interpolation [1]. "
        "A cross-encoder model then re-scores top candidates [2]."
    )

    tracked = tracker.track(answer_text, [r1, r2])

    assert len(tracked.citations) == 2
    assert tracked.citations[0].index == 1
    assert tracked.citations[0].source == "search.md"
    assert tracked.citations[0].page_number == 4
    assert tracked.citations[1].index == 2
    assert tracked.citations[1].source == "rerank.md"

    # Check claim attribution
    assert len(tracked.claim_citations) == 2
    claim1 = tracked.claim_citations[0]
    assert "hybrid search" in claim1.claim
    assert claim1.citation_indices == [1]
    assert "Hybrid search merges" in claim1.citations[0].quote

    assert tracked.coverage_ratio == 1.0


def test_citation_tracker_empty():
    tracker = CitationTracker()
    tracked = tracker.track("", [])
    assert tracked.raw_text == ""
    assert len(tracked.citations) == 0
