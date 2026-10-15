"""Hierarchical Parent-Document retrieval (Small-to-Big chunking)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from signalrag.models.chunk import Chunk, ChunkMetadata
from signalrag.models.retrieval import SearchResult


class ParentChunk(BaseModel):
    """Parent document section providing comprehensive surrounding context."""

    id: str
    document_id: str
    text: str
    child_chunk_ids: list[str] = Field(default_factory=list)
    metadata: ChunkMetadata


class HierarchicalRetriever:
    """Indexes fine-grained child chunks for search and expands to parent chunks for LLM generation."""

    def __init__(self, parent_size: int = 600, child_size: int = 150):
        self.parent_size = parent_size
        self.child_size = child_size
        self._parent_store: dict[str, ParentChunk] = {}
        self._child_to_parent: dict[str, str] = {}

    def build_hierarchy(
        self,
        document_id: str,
        text: str,
        source: str = "doc",
    ) -> tuple[list[ParentChunk], list[Chunk]]:
        """Split text into parent chunks and subdivide into child chunks."""
        parents: list[ParentChunk] = []
        children: list[Chunk] = []

        # 1. Break into parents
        for p_idx, p_start in enumerate(range(0, len(text), self.parent_size)):
            p_text = text[p_start : p_start + self.parent_size].strip()
            if not p_text:
                continue

            p_id = f"parent_{document_id}_{p_idx}"
            p_meta = ChunkMetadata(
                document_id=document_id,
                chunk_index=p_idx,
                source=source,
                extra={"is_parent": True},
            )

            p_chunk = ParentChunk(
                id=p_id,
                document_id=document_id,
                text=p_text,
                metadata=p_meta,
            )

            # 2. Break parent into children
            child_ids: list[str] = []
            for c_idx, c_start in enumerate(range(0, len(p_text), self.child_size)):
                c_text = p_text[c_start : c_start + self.child_size].strip()
                if not c_text:
                    continue

                c_id = f"child_{p_id}_{c_idx}"
                c_meta = ChunkMetadata(
                    document_id=document_id,
                    chunk_index=c_idx,
                    source=source,
                    extra={"parent_id": p_id},
                )

                c_chunk = Chunk(
                    id=c_id,
                    document_id=document_id,
                    text=c_text,
                    metadata=c_meta,
                )

                children.append(c_chunk)
                child_ids.append(c_id)
                self._child_to_parent[c_id] = p_id

            p_chunk.child_chunk_ids = child_ids
            parents.append(p_chunk)
            self._parent_store[p_id] = p_chunk

        return parents, children

    def expand_results(self, child_results: list[SearchResult]) -> list[SearchResult]:
        """Convert matched child search results into deduplicated parent context results."""
        seen_parent_ids: set[str] = set()
        expanded: list[SearchResult] = []

        for cr in child_results:
            p_id = self._child_to_parent.get(cr.chunk.id)
            if not p_id or p_id in seen_parent_ids:
                continue

            parent = self._parent_store.get(p_id)
            if not parent:
                continue

            seen_parent_ids.add(p_id)
            # Reconstruct parent as standard Chunk
            parent_as_chunk = Chunk(
                id=parent.id,
                document_id=parent.document_id,
                text=parent.text,
                metadata=parent.metadata,
            )
            expanded.append(SearchResult(chunk=parent_as_chunk, score=cr.score))

        return expanded
