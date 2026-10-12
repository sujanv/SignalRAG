"""Tests for token budget management."""

from signalrag.generation.budget import TokenBudgetAllocator, estimate_token_count
from signalrag.models.chunk import Chunk, ChunkMetadata


def test_estimate_token_count():
    assert estimate_token_count("hello world") >= 1


def test_token_budget_allocator():
    allocator = TokenBudgetAllocator(
        max_context_tokens=1000,
        system_reserved_tokens=200,
        generation_reserved_tokens=300,
    )
    assert allocator.available_retrieval_budget == 500

    chunks = [
        Chunk(
            id=f"c_{i}",
            document_id="doc1",
            text="Sentence one about data engineering. Sentence two about vector pipelines. " * 5,
            metadata=ChunkMetadata(document_id="doc1", chunk_index=i, source="doc.txt"),
        )
        for i in range(10)
    ]

    res = allocator.allocate(chunks, query="data engineering")
    assert len(res.allocated_chunks) > 0
    assert len(res.allocated_chunks) < 10
    assert res.total_tokens_used <= allocator.available_retrieval_budget
    assert res.dropped_chunk_count > 0
