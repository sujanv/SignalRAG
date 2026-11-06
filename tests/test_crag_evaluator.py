"""Tests for Corrective RAG (CRAG)."""

from signalrag.generation.crag import CRAGEvaluator
from signalrag.models.chunk import Chunk, ChunkMetadata


def test_crag_evaluator_grades():
    evaluator = CRAGEvaluator(high_threshold=0.3, low_threshold=0.1)

    chunks = [
        Chunk(
            id="c1",
            document_id="d1",
            text="Vector search and hybrid retrieval systems in SignalRAG.",
            metadata=ChunkMetadata(document_id="d1", chunk_index=0, source="doc1"),
        )
    ]

    # 1. Correct test
    correct_res = evaluator.evaluate_retrieval("vector search hybrid", chunks)
    assert correct_res.grade == "correct"
    assert correct_res.confidence >= 0.3
    assert "c1" in correct_res.retained_chunk_ids

    # 2. Incorrect test
    incorrect_res = evaluator.evaluate_retrieval("biology photosynthesis chlorophyll", chunks)
    assert incorrect_res.grade == "incorrect"
    assert incorrect_res.action == "fallback_web_synthesis"
    assert len(incorrect_res.retained_chunk_ids) == 0
    assert "photosynthesis" in incorrect_res.synthesized_context

    # 3. Empty chunks test
    empty_res = evaluator.evaluate_retrieval("empty test", [])
    assert empty_res.grade == "incorrect"
    assert empty_res.action == "fallback_web_synthesis"
