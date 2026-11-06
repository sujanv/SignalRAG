"""Tests for Explainable Retrieval Attribution."""

from signalrag.models.chunk import Chunk, ChunkMetadata
from signalrag.retrieval.explainer import RetrievalExplainer


def test_retrieval_explainer():
    chunk = Chunk(
        id="c_explain",
        document_id="d1",
        text="SignalRAG delivers high performance vector search with low latency indices.",
        metadata=ChunkMetadata(document_id="d1", chunk_index=0, source="test.txt"),
    )

    explainer = RetrievalExplainer()
    exp = explainer.explain("vector search latency", chunk, dense_similarity=0.88)

    assert exp.chunk_id == "c_explain"
    assert "vector" in exp.matching_terms
    assert "search" in exp.matching_terms
    assert "latency" in exp.matching_terms
    assert exp.lexical_overlap == 1.0
    assert exp.overall_score > 0.8
    assert "c_explain" in exp.summary
