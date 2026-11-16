"""Tests for Semantic Cache Invalidation."""

from signalrag.models.chunk import Chunk, ChunkMetadata
from signalrag.models.retrieval import SearchResult
from signalrag.retrieval.cache_invalidation import CacheInvalidationLedger
from signalrag.retrieval.semantic_cache import SemanticQueryCache


def test_cache_invalidation_lifecycle():
    cache = SemanticQueryCache()
    ledger = CacheInvalidationLedger()

    c1 = Chunk(
        id="doc_hr_c1",
        document_id="doc_hr",
        text="Vacation days are 20 per year.",
        metadata=ChunkMetadata(document_id="doc_hr", chunk_index=0, source="hr.txt"),
    )
    c2 = Chunk(
        id="doc_eng_c1",
        document_id="doc_eng",
        text="Deployments require two reviews.",
        metadata=ChunkMetadata(document_id="doc_eng", chunk_index=0, source="eng.txt"),
    )

    res_hr = [SearchResult(chunk=c1, score=0.9)]
    res_eng = [SearchResult(chunk=c2, score=0.9)]

    q_hr = "How many vacation days?"
    q_eng = "How to deploy code?"

    cache.set(q_hr, [1.0, 0.0], res_hr)
    cache.set(q_eng, [0.0, 1.0], res_eng)

    ledger.register_dependency("doc_hr", q_hr)
    ledger.register_dependency("doc_eng", q_eng)

    assert len(cache._entries) == 2

    report = ledger.invalidate_document("doc_hr", cache)
    assert report.document_id == "doc_hr"
    assert q_hr in report.purged_keys
    assert len(cache._entries) == 1

    # Eng query remains valid
    eng_hit = cache.get([0.0, 1.0])
    assert eng_hit is not None
