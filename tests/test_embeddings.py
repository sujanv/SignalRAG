"""Tests for embedding services and vector math."""

import math

from signalrag.core.config import EmbeddingConfig
from signalrag.embeddings import (
    DeterministicHashEmbeddingService,
    cosine_similarity,
    create_embedding_service,
)
from signalrag.models.chunk import Chunk


def test_hash_embedding_dimension_and_norm():
    service = DeterministicHashEmbeddingService(dimension=128)
    assert service.dimension == 128

    vec = service.embed_text("SignalRAG retrieval evaluation")
    assert len(vec) == 128

    # Verify unit norm
    norm = math.sqrt(sum(x * x for x in vec))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)


def test_cosine_similarity_semantic_behavior():
    service = DeterministicHashEmbeddingService(dimension=256)

    text_a = "Deep neural networks for natural language processing and transformer models"
    text_b = "Transformer neural networks and deep language models"
    text_c = "Delicious Italian pasta recipe with tomatoes, olive oil and garlic"

    vec_a = service.embed_text(text_a)
    vec_b = service.embed_text(text_b)
    vec_c = service.embed_text(text_c)

    sim_ab = cosine_similarity(vec_a, vec_b)
    sim_ac = cosine_similarity(vec_a, vec_c)

    # Identical vector similarity should be 1.0
    assert math.isclose(cosine_similarity(vec_a, vec_a), 1.0, rel_tol=1e-5)

    # Related texts should have higher similarity than unrelated text
    assert sim_ab > sim_ac
    assert sim_ab > 0.4


def test_embed_chunks_in_place():
    service = DeterministicHashEmbeddingService(dimension=64)
    chunk = Chunk.create(
        document_id="doc1",
        chunk_index=0,
        text="Sample text for embedding chunk test.",
        source="sample.txt",
    )
    assert chunk.embedding is None

    service.embed_chunks([chunk])
    assert chunk.embedding is not None
    assert len(chunk.embedding) == 64


def test_create_embedding_service_factory():
    cfg = EmbeddingConfig(provider="mock", dimension=128)
    svc = create_embedding_service(cfg)
    assert isinstance(svc, DeterministicHashEmbeddingService)
    assert svc.dimension == 128
