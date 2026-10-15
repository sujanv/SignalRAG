"""Concurrent multi-threaded batch indexing pipeline for enterprise ingestion."""

from __future__ import annotations

import concurrent.futures
import time

from pydantic import BaseModel

from signalrag.chunking.recursive import RecursiveCharacterChunker
from signalrag.embeddings.base import BaseEmbeddingService
from signalrag.indexing.vector_store import BaseVectorStore
from signalrag.models.document import Document


class BatchIndexReport(BaseModel):
    """Execution telemetry for batch indexing run."""

    documents_processed: int
    chunks_created: int
    elapsed_seconds: float
    chunks_per_second: float


class ConcurrentBatchIndexer:
    """Dispatches document chunking and embedding across concurrent thread pools."""

    def __init__(self, max_workers: int = 4, batch_size: int = 16):
        self.max_workers = max_workers
        self.batch_size = batch_size

    def index_documents(
        self,
        documents: list[Document],
        chunker: RecursiveCharacterChunker,
        embedding_service: BaseEmbeddingService,
        vector_store: BaseVectorStore,
    ) -> BatchIndexReport:
        """Process documents concurrently and index into vector store in batched transactions."""
        t0 = time.perf_counter()

        # 1. Parallel chunking
        all_chunks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(chunker.chunk_document, doc) for doc in documents]
            for f in concurrent.futures.as_completed(futures):
                all_chunks.extend(f.result())

        # 2. Batched embedding
        for i in range(0, len(all_chunks), self.batch_size):
            batch = all_chunks[i : i + self.batch_size]
            embedding_service.embed_chunks(batch)
            vector_store.add_chunks(batch)

        elapsed = max(0.001, time.perf_counter() - t0)
        throughput = len(all_chunks) / elapsed

        return BatchIndexReport(
            documents_processed=len(documents),
            chunks_created=len(all_chunks),
            elapsed_seconds=round(elapsed, 3),
            chunks_per_second=round(throughput, 1),
        )
