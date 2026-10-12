"""Tests for Pseudo-Relevance Feedback and query expansion."""

from signalrag.models.chunk import Chunk, ChunkMetadata
from signalrag.models.retrieval import SearchResult
from signalrag.retrieval.expansion import PseudoRelevanceExpander


class DummyPipeline:
    def retrieve(self, query: str, top_k: int = 5):
        chunk = Chunk(
            id="c_feedback",
            document_id="doc_fb",
            text="Vector embeddings utilize high-dimensional cosine similarity and dense representations.",
            metadata=ChunkMetadata(document_id="doc_fb", chunk_index=0, source="doc.txt"),
        )
        return [SearchResult(chunk=chunk, score=0.9)]


def test_pseudo_relevance_expansion_terms():
    expander = PseudoRelevanceExpander(top_n_terms=2)
    chunks = [
        Chunk(
            id="c1",
            document_id="d1",
            text="Machine learning pipelines require scalable vector databases and approximate nearest neighbors.",
            metadata=ChunkMetadata(document_id="d1", chunk_index=0, source="d1.txt"),
        )
    ]
    res = expander.expand("machine learning", chunks)
    assert res.original_query == "machine learning"
    assert len(res.expansion_terms) == 2
    assert "machine" not in res.expansion_terms
    assert "learning" not in res.expansion_terms
    assert res.expanded_query.startswith("machine learning ")


def test_expand_empty_chunks():
    expander = PseudoRelevanceExpander()
    res = expander.expand("test query", [])
    assert res.expanded_query == "test query"
    assert len(res.expansion_terms) == 0


def test_retrieve_with_prf():
    pipeline = DummyPipeline()
    expander = PseudoRelevanceExpander(top_n_terms=2)
    exp_res, results = expander.retrieve_with_prf("search query", pipeline, top_k=2)
    assert len(results) == 1
    assert exp_res.original_query == "search query"
    assert len(exp_res.expansion_terms) > 0
