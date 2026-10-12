"""Semantic near-duplicate detection and chunk clustering."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from signalrag.models.chunk import Chunk


def _get_shingles(text: str, k: int = 3) -> set[str]:
    """Extract character k-shingles from text."""
    clean = re.sub(r"\s+", " ", text.lower().strip())
    if len(clean) < k:
        return {clean}
    return {clean[i : i + k] for i in range(len(clean) - k + 1)}


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity coefficient between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


class DedupReport(BaseModel):
    """Outcome of chunk deduplication."""

    original_count: int
    retained_count: int
    duplicates_removed: int
    clusters: list[list[str]] = Field(default_factory=list)


class SemanticChunkDeduplicator:
    """Detects and prunes redundant and near-duplicate chunks."""

    def __init__(self, similarity_threshold: float = 0.75, shingle_size: int = 3):
        self.similarity_threshold = similarity_threshold
        self.shingle_size = shingle_size

    def deduplicate(self, chunks: list[Chunk]) -> tuple[list[Chunk], DedupReport]:
        """Cluster near-duplicates and retain one representative chunk per cluster."""
        if not chunks:
            return [], DedupReport(
                original_count=0,
                retained_count=0,
                duplicates_removed=0,
                clusters=[],
            )

        shingle_cache = [_get_shingles(c.text, self.shingle_size) for c in chunks]

        clusters: list[list[int]] = []
        assigned = set()

        for i in range(len(chunks)):
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j in range(i + 1, len(chunks)):
                if j in assigned:
                    continue
                sim = jaccard_similarity(shingle_cache[i], shingle_cache[j])
                if sim >= self.similarity_threshold:
                    cluster.append(j)
                    assigned.add(j)
            clusters.append(cluster)

        # For each cluster, pick the longest chunk as representative
        retained: list[Chunk] = []
        cluster_id_lists: list[list[str]] = []

        for cluster_indices in clusters:
            best_idx = max(cluster_indices, key=lambda idx: len(chunks[idx].text))
            retained.append(chunks[best_idx])
            cluster_id_lists.append([chunks[idx].id for idx in cluster_indices])

        report = DedupReport(
            original_count=len(chunks),
            retained_count=len(retained),
            duplicates_removed=len(chunks) - len(retained),
            clusters=cluster_id_lists,
        )
        return retained, report
