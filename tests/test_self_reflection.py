"""Tests for Self-Reflective RAG (Self-RAG)."""

from signalrag.generation.self_reflection import SelfReflectiveEngine
from signalrag.models.chunk import Chunk, ChunkMetadata


def test_self_reflection_relevant():
    engine = SelfReflectiveEngine(relevance_threshold=0.4)
    chunks = [
        Chunk(
            id="c1",
            document_id="d1",
            text="Vector search uses cosine similarity across embedding spaces.",
            metadata=ChunkMetadata(document_id="d1", chunk_index=0, source="s1"),
        )
    ]
    dec = engine.grade_retrieval("vector search cosine similarity", chunks)
    assert dec.retrieval_grade == "RELEVANT"
    assert dec.action == "GENERATE"
    assert dec.needs_re_retrieval is False


def test_self_reflection_irrelevant():
    engine = SelfReflectiveEngine(relevance_threshold=0.5)
    chunks = [
        Chunk(
            id="c1",
            document_id="d1",
            text="Completely unrelated recipes for chocolate chip cookies.",
            metadata=ChunkMetadata(document_id="d1", chunk_index=0, source="s1"),
        )
    ]
    dec = engine.grade_retrieval("distributed consensus algorithms", chunks)
    assert dec.retrieval_grade == "IRRELEVANT"
    assert dec.action == "RE_RETRIEVE"
    assert dec.needs_re_retrieval is True
    assert dec.reformulated_query is not None
