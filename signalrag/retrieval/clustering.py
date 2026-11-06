"""Semantic clustering and topic discovery for large-scale chunk corpuses."""

from __future__ import annotations

import collections
import re

from pydantic import BaseModel, Field

from signalrag.embeddings.base import cosine_similarity
from signalrag.models.chunk import Chunk


class ClusterCentroid(BaseModel):
    """Cluster representation including semantic centroid and keyword topic name."""

    cluster_id: int
    topic_name: str
    centroid: list[float]
    chunk_ids: list[str] = Field(default_factory=list)
    size: int = 0


class SemanticClusterEngine:
    """Partitions chunk embeddings into semantic topic clusters using k-means centroid clustering."""

    def __init__(self, k: int = 3, max_iters: int = 10):
        self.k = k
        self.max_iters = max_iters

    def cluster_chunks(self, chunks: list[Chunk], k: int | None = None) -> list[ClusterCentroid]:
        """Cluster valid embedded chunks into k semantic partitions."""
        valid_chunks = [c for c in chunks if c.embedding is not None and len(c.embedding) > 0]
        if not valid_chunks:
            return []

        target_k = min(k or self.k, len(valid_chunks))
        dim = len(valid_chunks[0].embedding)  # type: ignore

        # Initialize centroids with initial chunk vectors
        centroids = [list(valid_chunks[i].embedding) for i in range(target_k)]  # type: ignore

        assignments: dict[int, list[Chunk]] = {i: [] for i in range(target_k)}

        for _ in range(self.max_iters):
            assignments = {i: [] for i in range(target_k)}
            for chunk in valid_chunks:
                # Find closest centroid
                best_sim = -1.0
                best_idx = 0
                for c_idx, c_vec in enumerate(centroids):
                    sim = cosine_similarity(chunk.embedding, c_vec)  # type: ignore
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = c_idx
                assignments[best_idx].append(chunk)

            # Recompute centroids
            for c_idx in range(target_k):
                cluster_points = assignments[c_idx]
                if not cluster_points:
                    continue
                new_vec = [0.0] * dim
                for pt in cluster_points:
                    for d in range(dim):
                        new_vec[d] += pt.embedding[d]  # type: ignore
                centroids[c_idx] = [val / len(cluster_points) for val in new_vec]

        # Name topics based on top keywords in cluster
        results: list[ClusterCentroid] = []
        for c_idx in range(target_k):
            cluster_points = assignments[c_idx]
            words = []
            for pt in cluster_points:
                words.extend(re.findall(r"\b[a-zA-Z]{4,}\b", pt.text.lower()))
            common = collections.Counter(words).most_common(2)
            topic_label = "-".join(w for w, _ in common) if common else f"topic-{c_idx}"

            results.append(
                ClusterCentroid(
                    cluster_id=c_idx,
                    topic_name=topic_label,
                    centroid=centroids[c_idx],
                    chunk_ids=[pt.id for pt in cluster_points],
                    size=len(cluster_points),
                )
            )

        return results

    def predict_cluster(self, query_vector: list[float], clusters: list[ClusterCentroid]) -> int:
        """Assign query to nearest topic cluster centroid."""
        if not clusters:
            return -1
        best_sim = -1.0
        best_id = clusters[0].cluster_id
        for cl in clusters:
            sim = cosine_similarity(query_vector, cl.centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = cl.cluster_id
        return best_id
