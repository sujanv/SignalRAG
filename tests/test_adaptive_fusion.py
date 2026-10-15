"""Tests for adaptive hybrid fusion weighting."""

from signalrag.models.chunk import Chunk, ChunkMetadata
from signalrag.models.retrieval import SearchResult
from signalrag.retrieval.adaptive_fusion import AdaptiveHybridFuser


def test_determine_weights():
    fuser = AdaptiveHybridFuser()

    kw_weights = fuser.determine_weights("ERR_404_TIMEOUT")
    assert kw_weights.bm25_weight > kw_weights.vector_weight
    assert kw_weights.query_type == "keyword"

    concept_weights = fuser.determine_weights(
        "Why is high dimensional vector search effective for retrieval?"
    )
    assert concept_weights.vector_weight > concept_weights.bm25_weight
    assert concept_weights.query_type == "conceptual"


def test_score_normalization_and_fusion():
    fuser = AdaptiveHybridFuser()
    c1 = Chunk(
        id="c1",
        document_id="d1",
        text="chunk 1",
        metadata=ChunkMetadata(document_id="d1", chunk_index=0, source="s1"),
    )
    c2 = Chunk(
        id="c2",
        document_id="d2",
        text="chunk 2",
        metadata=ChunkMetadata(document_id="d2", chunk_index=0, source="s2"),
    )

    vec_res = [SearchResult(chunk=c1, score=0.9), SearchResult(chunk=c2, score=0.3)]
    bm25_res = [SearchResult(chunk=c2, score=10.0), SearchResult(chunk=c1, score=2.0)]

    fused = fuser.fuse("test query", vec_res, bm25_res, top_k=2)
    assert len(fused) == 2
    assert fused[0].score >= fused[1].score
