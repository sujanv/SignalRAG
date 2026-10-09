"""Tests for hybrid retrieval merging semantic and BM25 search."""

import pytest

from signalrag.embeddings import DeterministicHashEmbeddingService
from signalrag.indexing import MemoryVectorStore
from signalrag.models.chunk import Chunk
from signalrag.retrieval import BM25Retriever, HybridRetriever, SemanticRetriever


@pytest.fixture
def hybrid_setup():
    store = MemoryVectorStore()
    emb_service = DeterministicHashEmbeddingService(dimension=64)

    # c1 has exact identifier 'ERR_502_BAD_GATEWAY' (favors BM25)
    c1 = Chunk.create(
        "d1",
        0,
        "Server response failure code ERR_502_BAD_GATEWAY downstream upstream proxy.",
        "proxy.txt",
    )
    # c2 has semantic paraphrasing of networking and latency (favors Semantic)
    c2 = Chunk.create(
        "d2", 0, "Network transmission bottlenecks and packet roundtrip delay issues.", "net.txt"
    )
    # c3 is unrelated
    c3 = Chunk.create("d3", 0, "Baking bread with yeast and whole wheat grain flour.", "recipe.txt")

    chunks = [c1, c2, c3]
    emb_service.embed_chunks(chunks)
    store.add_chunks(chunks)

    semantic = SemanticRetriever(store, emb_service)
    bm25 = BM25Retriever(chunks)
    return semantic, bm25


def test_hybrid_retriever_linear_fusion(hybrid_setup):
    semantic, bm25 = hybrid_setup
    hybrid = HybridRetriever(semantic, bm25, alpha=0.5, fusion_method="linear")

    results = hybrid.retrieve("ERR_502_BAD_GATEWAY", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk.metadata.source == "proxy.txt"
    assert results[0].retrieval_method == "hybrid"


def test_hybrid_retriever_rrf_fusion(hybrid_setup):
    semantic, bm25 = hybrid_setup
    hybrid = HybridRetriever(semantic, bm25, alpha=0.5, fusion_method="rrf")

    results = hybrid.retrieve("network packet delay bottlenecks", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk.metadata.source == "net.txt" or results[0].rank == 1


def test_hybrid_alpha_extremes(hybrid_setup):
    semantic, bm25 = hybrid_setup

    pure_vec = HybridRetriever(semantic, bm25, alpha=1.0)
    pure_bm25 = HybridRetriever(semantic, bm25, alpha=0.0)

    res_v = pure_vec.retrieve("transmission packet delay", top_k=1)
    res_b = pure_bm25.retrieve("ERR_502_BAD_GATEWAY", top_k=1)

    assert res_v[0].retrieval_method == "vector"
    assert res_b[0].retrieval_method == "bm25"


def test_hybrid_invalid_alpha(hybrid_setup):
    semantic, bm25 = hybrid_setup
    with pytest.raises(ValueError):
        HybridRetriever(semantic, bm25, alpha=1.5)
