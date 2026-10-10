"""Tests for MultiHop query decomposition and retrieval."""

from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult
from signalrag.retrieval.decomposer import MultiHopRetriever, QueryDecomposer


class MockPipeline:
    def retrieve(self, query: str, top_k: int = 3):
        c = Chunk(
            id=f"chunk_{hash(query) % 1000}",
            document_id="doc_test",
            text=f"Evidence for query: {query}",
        )
        return [SearchResult(chunk=c, score=0.9)]


def test_query_decomposer_simple():
    decomposer = QueryDecomposer()
    plan = decomposer.decompose("What is BM25?")
    assert plan.is_complex is False
    assert len(plan.sub_queries) == 1
    assert plan.sub_queries[0].query == "What is BM25?"


def test_query_decomposer_compound():
    decomposer = QueryDecomposer()
    plan = decomposer.decompose("What is BM25 lexical ranking and how does vector search work?")
    assert plan.is_complex is True
    assert len(plan.sub_queries) == 2


def test_query_decomposer_comparative():
    decomposer = QueryDecomposer()
    plan = decomposer.decompose("Compare BM25 versus Dense Embeddings")
    assert plan.is_complex is True
    assert len(plan.sub_queries) == 2
    assert any("BM25" in sq.query for sq in plan.sub_queries)
    assert any("Dense Embeddings" in sq.query for sq in plan.sub_queries)


def test_multihop_retriever_execution():
    pipeline = MockPipeline()
    retriever = MultiHopRetriever(pipeline=pipeline)
    trace = retriever.retrieve("Compare BM25 and vector search")

    assert trace.plan.is_complex is True
    assert len(trace.sub_results) == 2
    assert len(trace.merged_results) >= 1
    assert trace.total_latency_ms >= 0.0
