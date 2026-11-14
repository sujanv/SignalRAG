"""Tests for Speculative RAG Drafting and Verification."""

from signalrag.generation.speculative import SpeculativeRAGEngine
from signalrag.models.chunk import Chunk, ChunkMetadata


def test_speculative_rag_draft_and_verify():
    engine = SpeculativeRAGEngine()

    chunk = Chunk(
        id="c1",
        document_id="d1",
        text="Consistent hash rings allow horizontal distribution of vector index shards.",
        metadata=ChunkMetadata(document_id="d1", chunk_index=0, source="spec.txt"),
    )

    draft = engine.draft("Explain hash rings", [chunk])
    assert "preliminary" in draft.draft_answer.lower()
    assert draft.proposed_chunk_ids == ["c1"]

    verified = engine.verify("Explain hash rings", draft, [chunk])
    assert verified.grounding_confidence >= 0.4
    assert len(verified.verified_answer) > 10
    assert verified.total_latency_ms >= 0.0


def test_speculative_rag_empty_chunks():
    engine = SpeculativeRAGEngine()
    draft = engine.draft("Empty query", [])
    assert "insufficient" in draft.draft_answer.lower()

    verified = engine.verify("Empty query", draft, [])
    assert "refusal" in verified.verified_answer.lower()
