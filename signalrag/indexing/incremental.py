"""Incremental indexing subsystem tracking content hashes to prevent redundant processing."""

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from signalrag.indexing.pipeline import IndexingPipeline
from signalrag.models.document import Document


class DocumentLedgerRecord(BaseModel):
    """Ledger entry storing document hashing and indexed chunk IDs."""

    doc_id: str
    source: str
    content_hash: str
    chunk_ids: list[str] = Field(default_factory=list)
    last_indexed_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DocumentLedger(BaseModel):
    """Persistent ledger tracking indexed documents and hashes."""

    records: dict[str, DocumentLedgerRecord] = Field(default_factory=dict)

    @classmethod
    def load_from_file(cls, ledger_path: str | Path) -> "DocumentLedger":
        path = Path(ledger_path)
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(**data)
        except Exception:
            return cls()

    def save_to_file(self, ledger_path: str | Path) -> None:
        path = Path(ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2), encoding="utf-8")


@dataclass
class IncrementalIndexingResult:
    """Summary of an incremental indexing run."""

    total_examined: int
    added_documents: int
    updated_documents: int
    skipped_documents: int
    deleted_documents: int
    added_chunks: int
    duration_seconds: float
    stats: dict[str, Any] = field(default_factory=dict)


class IncrementalIndexer:
    """Performs hash-checked incremental indexing over documents."""

    def __init__(
        self,
        pipeline: IndexingPipeline,
        ledger_path: str | Path = "./storage/vector_store/ledger.json",
    ) -> None:
        self.pipeline = pipeline
        self.ledger_path = Path(ledger_path)
        self.ledger = DocumentLedger.load_from_file(self.ledger_path)

    def index(self, documents: list[Document], force: bool = False) -> IncrementalIndexingResult:
        """Incrementally index documents, skipping unchanged ones and replacing updated ones."""
        start_time = time.perf_counter()

        added_docs = 0
        updated_docs = 0
        skipped_docs = 0
        total_new_chunks = 0

        docs_to_process: list[Document] = []
        old_chunk_ids_to_remove: list[str] = []

        for doc in documents:
            source_key = doc.metadata.source
            existing = self.ledger.records.get(source_key)

            if existing is None:
                # Completely new document
                added_docs += 1
                docs_to_process.append(doc)
            elif force or existing.content_hash != doc.metadata.content_hash:
                # Content has changed
                updated_docs += 1
                old_chunk_ids_to_remove.extend(existing.chunk_ids)
                docs_to_process.append(doc)
            else:
                # Unchanged document
                skipped_docs += 1

        # 1. Clean up stale chunks from updated documents
        if old_chunk_ids_to_remove:
            self.pipeline.vector_store.delete(old_chunk_ids_to_remove)

        # 2. Process and embed changed/new documents
        if docs_to_process:
            for doc in docs_to_process:
                doc_chunks = self.pipeline.chunker.chunk_document(doc)
                if doc_chunks:
                    self.pipeline.embedding_service.embed_chunks(doc_chunks)
                    self.pipeline.vector_store.add_chunks(doc_chunks)
                    total_new_chunks += len(doc_chunks)

                # Update ledger record
                self.ledger.records[doc.metadata.source] = DocumentLedgerRecord(
                    doc_id=doc.id,
                    source=doc.metadata.source,
                    content_hash=doc.metadata.content_hash or "",
                    chunk_ids=[c.id for c in doc_chunks],
                    last_indexed_at=datetime.now(UTC).isoformat(),
                )

            # Persist updated ledger
            self.ledger.save_to_file(self.ledger_path)
            self.pipeline.vector_store.save()

        duration = time.perf_counter() - start_time
        return IncrementalIndexingResult(
            total_examined=len(documents),
            added_documents=added_docs,
            updated_documents=updated_docs,
            skipped_documents=skipped_docs,
            deleted_documents=0,
            added_chunks=total_new_chunks,
            duration_seconds=round(duration, 4),
            stats={
                "total_ledger_documents": len(self.ledger.records),
                "vector_store_chunks": self.pipeline.vector_store.count(),
            },
        )
