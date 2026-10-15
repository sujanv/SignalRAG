"""Tests for GraphRAG knowledge graph retrieval."""

from signalrag.models.chunk import Chunk, ChunkMetadata
from signalrag.retrieval.graph import KnowledgeGraphRetriever


def test_knowledge_graph_extraction_and_traversal():
    kg = KnowledgeGraphRetriever()

    chunks = [
        Chunk(
            id="c1",
            document_id="d1",
            text="SignalRAG utilizes BM25 for keyword scoring.",
            metadata=ChunkMetadata(document_id="d1", chunk_index=0, source="s1"),
        ),
        Chunk(
            id="c2",
            document_id="d2",
            text="BM25 implements Lucene ranking equations.",
            metadata=ChunkMetadata(document_id="d2", chunk_index=1, source="s2"),
        ),
    ]

    triples = kg.index_chunks(chunks)
    assert triples >= 2
    assert "SignalRAG" in kg.entities
    assert "BM25" in kg.entities

    # Subgraph retrieval
    ctx = kg.retrieve_subgraph("How does SignalRAG operate?")
    assert len(ctx.entities) >= 1
    assert any(e.name == "SignalRAG" for e in ctx.entities)
    assert len(ctx.relations) >= 1
    assert "c1" in ctx.connected_chunk_ids
