"""Tests for HyDE (Hypothetical Document Embeddings)."""

from signalrag.embeddings.hash_embedding import DeterministicHashEmbeddingService
from signalrag.indexing.memory_store import MemoryVectorStore
from signalrag.models.chunk import Chunk, ChunkMetadata
from signalrag.retrieval.hyde import HyDERetriever


def test_hyde_retriever_flow():
    emb = DeterministicHashEmbeddingService(dimension=16)
    store = MemoryVectorStore()

    chunk = Chunk(
        id="c1",
        document_id="d1",
        text="SignalRAG implements dense vector search and BM25.",
        metadata=ChunkMetadata(document_id="d1", chunk_index=0, source="s1"),
        embedding=emb.embed_text("SignalRAG implements dense vector search and BM25."),
    )
    store.add_chunks([chunk])

    hyde = HyDERetriever(vector_store=store, embedding_service=emb)
    hypo = hyde.generate_hypothetical_document("How does SignalRAG work?")

    assert len(hypo.hypothetical_passage) > 20
    assert "SignalRAG" in hypo.hypothetical_passage

    hits = hyde.retrieve("How does SignalRAG work?", top_k=1)
    assert len(hits) == 1
    assert hits[0].chunk.id == "c1"
