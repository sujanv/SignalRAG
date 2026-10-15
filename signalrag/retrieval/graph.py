"""Graph-based retrieval (GraphRAG) mapping entity-relationship networks across documents."""

from __future__ import annotations

import collections
import re

from pydantic import BaseModel, Field

from signalrag.models.chunk import Chunk


class EntityNode(BaseModel):
    """Knowledge graph entity node."""

    name: str
    entity_type: str = "CONCEPT"
    chunk_ids: list[str] = Field(default_factory=list)


class RelationEdge(BaseModel):
    """Knowledge graph directed relationship edge."""

    source: str
    target: str
    relation: str
    weight: float = 1.0
    evidence_chunk_id: str | None = None


class GraphContext(BaseModel):
    """Structured graph retrieval context supporting generation."""

    entities: list[EntityNode] = Field(default_factory=list)
    relations: list[RelationEdge] = Field(default_factory=list)
    connected_chunk_ids: list[str] = Field(default_factory=list)
    subgraph_text: str = ""


class KnowledgeGraphRetriever:
    """Extracts, indexes, and traverses entity-relationship graphs for associative multi-hop retrieval."""

    RELATION_PATTERNS = [
        (
            re.compile(r"([A-Z][a-zA-Z0-9_]+)\s+(?:is a|is an|acts as)\s+([A-Z][a-zA-Z0-9_]+)"),
            "IS_A",
        ),
        (
            re.compile(r"([A-Z][a-zA-Z0-9_]+)\s+(?:uses|utilizes|employs)\s+([A-Z][a-zA-Z0-9_]+)"),
            "USES",
        ),
        (
            re.compile(
                r"([A-Z][a-zA-Z0-9_]+)\s+(?:connects to|interacts with|integrates with)\s+([A-Z][a-zA-Z0-9_]+)"
            ),
            "INTEGRATES_WITH",
        ),
        (
            re.compile(
                r"([A-Z][a-zA-Z0-9_]+)\s+(?:implements|extends|builds upon)\s+([A-Z][a-zA-Z0-9_]+)"
            ),
            "IMPLEMENTS",
        ),
    ]

    def __init__(self):
        self.entities: dict[str, EntityNode] = {}
        self.adj_list: dict[str, list[RelationEdge]] = collections.defaultdict(list)

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """Extract entities and relation triples from chunks and index into the knowledge graph."""
        triples_added = 0
        for chunk in chunks:
            text = chunk.text
            # Extract relation patterns
            for pattern, rel_type in self.RELATION_PATTERNS:
                for match in pattern.finditer(text):
                    src, tgt = match.group(1), match.group(2)
                    self._ensure_entity(src, chunk.id)
                    self._ensure_entity(tgt, chunk.id)

                    edge = RelationEdge(
                        source=src,
                        target=tgt,
                        relation=rel_type,
                        weight=1.0,
                        evidence_chunk_id=chunk.id,
                    )
                    self.adj_list[src].append(edge)
                    triples_added += 1
        return triples_added

    def _ensure_entity(self, name: str, chunk_id: str) -> None:
        if name not in self.entities:
            self.entities[name] = EntityNode(name=name, chunk_ids=[chunk_id])
        elif chunk_id not in self.entities[name].chunk_ids:
            self.entities[name].chunk_ids.append(chunk_id)

    def retrieve_subgraph(self, query: str, max_depth: int = 2) -> GraphContext:
        """Find query-mentioned entities and traverse their local neighborhood subgraph."""
        matched_roots = [
            name
            for name in self.entities
            if re.search(r"\b" + re.escape(name) + r"\b", query, re.IGNORECASE)
        ]

        if not matched_roots:
            return GraphContext()

        visited_entities: set[str] = set()
        collected_relations: list[RelationEdge] = []
        collected_chunk_ids: set[str] = set()

        queue: collections.deque[tuple[str, int]] = collections.deque(
            [(root, 0) for root in matched_roots]
        )

        while queue:
            curr_entity, depth = queue.popleft()
            if curr_entity in visited_entities or depth > max_depth:
                continue
            visited_entities.add(curr_entity)

            node = self.entities.get(curr_entity)
            if node:
                collected_chunk_ids.update(node.chunk_ids)

            for edge in self.adj_list.get(curr_entity, []):
                collected_relations.append(edge)
                if edge.evidence_chunk_id:
                    collected_chunk_ids.add(edge.evidence_chunk_id)
                if edge.target not in visited_entities and depth + 1 <= max_depth:
                    queue.append((edge.target, depth + 1))

        # Build textual representation
        lines = [f"({e.source}) -[{e.relation}]-> ({e.target})" for e in collected_relations]
        subgraph_text = "\n".join(lines)

        return GraphContext(
            entities=[self.entities[name] for name in visited_entities if name in self.entities],
            relations=collected_relations,
            connected_chunk_ids=list(collected_chunk_ids),
            subgraph_text=subgraph_text,
        )
