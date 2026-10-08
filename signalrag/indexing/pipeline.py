"""Unified end-to-end indexing pipeline: Documents -> Chunk -> Embed -> Index."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from signalrag.chunking.base import BaseChunker
from signalrag.chunking.recursive import RecursiveCharacterChunker
from signalrag.core.config import Settings, get_settings
from signalrag.embeddings.base import BaseEmbeddingService
from signalrag.embeddings.factory import create_embedding_service
from signalrag.indexing.memory_store import MemoryVectorStore
from signalrag.indexing.vector_store import BaseVectorStore
from signalrag.ingestion.directory_loader import DirectoryLoader
from signalrag.ingestion.pdf_loader import PDFLoader
from signalrag.ingestion.text_loader import TextLoader
from signalrag.models.document import Document
from signalrag.models.retrieval import SearchResult


@dataclass
class IndexingResult:
    """Summary of an indexing execution run."""

    total_documents: int
    total_chunks: int
    duration_seconds: float
    document_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class IndexingPipeline:
    """Coordinates document ingestion, chunking, embedding generation, and vector store persistence."""

    def __init__(
        self,
        chunker: BaseChunker | None = None,
        embedding_service: BaseEmbeddingService | None = None,
        vector_store: BaseVectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self.chunker = chunker or RecursiveCharacterChunker(
            chunk_size=cfg.chunking.chunk_size,
            chunk_overlap=cfg.chunking.chunk_overlap,
            min_chunk_size=cfg.chunking.min_chunk_size,
        )
        self.embedding_service = embedding_service or create_embedding_service(cfg.embedding)
        self.vector_store = vector_store or MemoryVectorStore(
            storage_path=cfg.vector_store.path / "index.json"
        )

    def index_documents(self, documents: list[Document]) -> IndexingResult:
        """Process documents through chunking, embedding, and indexing."""
        start_time = time.perf_counter()
        if not documents:
            return IndexingResult(total_documents=0, total_chunks=0, duration_seconds=0.0)

        # 1. Chunk documents
        all_chunks = []
        doc_ids = []
        for doc in documents:
            doc_ids.append(doc.id)
            doc_chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(doc_chunks)

        # 2. Compute embeddings
        if all_chunks:
            self.embedding_service.embed_chunks(all_chunks)

        # 3. Store in vector database
        added_count = self.vector_store.add_chunks(all_chunks)

        duration = time.perf_counter() - start_time
        return IndexingResult(
            total_documents=len(documents),
            total_chunks=added_count,
            duration_seconds=round(duration, 4),
            document_ids=doc_ids,
            chunk_ids=[c.id for c in all_chunks],
            metadata={
                "embedding_dimension": self.embedding_service.dimension,
                "store_total_chunks": self.vector_store.count(),
            },
        )

    def index_path(self, path: str | Path, recursive: bool = True) -> IndexingResult:
        """Ingest all documents at path and index them."""
        target = Path(path).resolve()
        if not target.exists():
            raise FileNotFoundError(f"Path does not exist: {target}")

        if target.is_file():
            suffix = target.suffix.lower()
            if suffix == ".pdf":
                loader = PDFLoader(target)
            else:
                loader = TextLoader(target)
            docs = loader.load()
        else:
            loader = DirectoryLoader(target, recursive=recursive)
            docs = loader.load()

        return self.index_documents(docs)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Query the vector index directly using embedded query vector."""
        if not query.strip():
            return []

        query_vector = self.embedding_service.embed_text(query)
        return self.vector_store.search(query_vector, top_k=top_k, filters=filters)
