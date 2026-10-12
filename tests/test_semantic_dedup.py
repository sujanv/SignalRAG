"""Tests for chunk deduplication and clustering."""

from signalrag.chunking.dedup import SemanticChunkDeduplicator, jaccard_similarity
from signalrag.models.chunk import Chunk, ChunkMetadata


def test_jaccard_similarity():
    s1 = {"abc", "def"}
    s2 = {"abc", "def"}
    assert jaccard_similarity(s1, s2) == 1.0
    assert jaccard_similarity(s1, set()) == 0.0


def test_chunk_deduplication():
    chunks = [
        Chunk(
            id="c1",
            document_id="d1",
            text="SignalRAG hybrid retrieval fuses dense vector embeddings and BM25 keywords.",
            metadata=ChunkMetadata(document_id="d1", chunk_index=0, source="s1"),
        ),
        Chunk(
            id="c2",
            document_id="d2",
            text="SignalRAG hybrid retrieval fuses dense vector embeddings and BM25 keywords exactly.",
            metadata=ChunkMetadata(document_id="d2", chunk_index=1, source="s2"),
        ),
        Chunk(
            id="c3",
            document_id="d3",
            text="Completely different passage about astronomical observations.",
            metadata=ChunkMetadata(document_id="d3", chunk_index=2, source="s3"),
        ),
    ]

    dedup = SemanticChunkDeduplicator(similarity_threshold=0.7)
    retained, report = dedup.deduplicate(chunks)

    assert report.original_count == 3
    assert report.retained_count == 2
    assert report.duplicates_removed == 1
    assert len(retained) == 2
