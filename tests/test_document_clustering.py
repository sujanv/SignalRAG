"""Tests for Semantic Document Clustering."""

from signalrag.embeddings.hash_embedding import DeterministicHashEmbeddingService
from signalrag.models.chunk import Chunk, ChunkMetadata
from signalrag.retrieval.clustering import SemanticClusterEngine


def test_semantic_cluster_engine():
    emb = DeterministicHashEmbeddingService(dimension=16)
    engine = SemanticClusterEngine(k=2, max_iters=5)

    texts = [
        "Machine learning neural network models.",
        "Deep learning backpropagation algorithms.",
        "Financial portfolio tax accounting and investments.",
        "Corporate bonds and banking revenue statements.",
    ]

    chunks = [
        Chunk(
            id=f"c{i}",
            document_id=f"d{i}",
            text=t,
            metadata=ChunkMetadata(document_id=f"d{i}", chunk_index=0, source="test"),
            embedding=emb.embed_text(t),
        )
        for i, t in enumerate(texts)
    ]

    clusters = engine.cluster_chunks(chunks, k=2)
    assert len(clusters) == 2
    total_assigned = sum(c.size for c in clusters)
    assert total_assigned == 4

    # Prediction test
    query_vec = emb.embed_text("neural network training")
    assigned_c = engine.predict_cluster(query_vec, clusters)
    assert assigned_c in [0, 1]
