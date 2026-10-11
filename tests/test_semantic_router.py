"""Tests for query routing."""

from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult
from signalrag.retrieval.router import QueryRouter


class MockPipeline:
    bm25_retriever = None
    semantic_retriever = None

    def retrieve(self, query: str, top_k: int = 5):
        return [SearchResult(chunk=Chunk(id="c1", document_id="d1", text="test"), score=1.0)]


def test_query_router_rules():
    router = QueryRouter()

    # Comparative
    dec_comp = router.route("Compare Vector DB versus Inverted Index")
    assert dec_comp.target_strategy == "multihop"
    assert dec_comp.confidence > 0.8

    # Keyword / exact identifier
    dec_kw = router.route("ERR_CONN_REFUSED 502")
    assert dec_kw.target_strategy == "bm25"

    # Conceptual
    dec_concept = router.route("What is the core architecture of SignalRAG?")
    assert dec_concept.target_strategy == "hybrid"


def test_execute_routed_query():
    router = QueryRouter()
    pipeline = MockPipeline()
    dec, res = router.execute_routed_query("Why is RAG popular?", pipeline, top_k=2)
    assert dec.target_strategy == "hybrid"
    assert len(res) == 1
