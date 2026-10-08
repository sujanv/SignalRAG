"""Sparse lexical retriever based on BM25+ ranking."""

import re
from typing import Any, Literal

import numpy as np
from rank_bm25 import BM25Okapi, BM25Plus

from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult
from signalrag.retrieval.base import BaseRetriever


def default_tokenizer(text: str) -> list[str]:
    """Tokenize and normalize text into lowercase word tokens."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever(BaseRetriever):
    """Lexical keyword retriever implementing BM25+ / BM25Okapi scoring with score normalization."""

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        algorithm: Literal["bm25+", "okapi"] = "bm25+",
        min_score: float = 0.0,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.algorithm = algorithm
        self.min_score = min_score
        self.chunks: list[Chunk] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Plus | BM25Okapi | None = None

        if chunks:
            self.index_chunks(chunks)

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Build the BM25 inverted index from a list of Chunks."""
        self.chunks = list(chunks)
        self._corpus_tokens = [default_tokenizer(c.text) for c in self.chunks]
        if self._corpus_tokens:
            if self.algorithm == "okapi":
                self._bm25 = BM25Okapi(self._corpus_tokens, k1=self.k1, b=self.b)
            else:
                self._bm25 = BM25Plus(self._corpus_tokens, k1=self.k1, b=self.b, delta=self.delta)
        else:
            self._bm25 = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add additional chunks and re-index."""
        self.index_chunks(self.chunks + chunks)

    def _matches_filters(self, chunk: Chunk, filters: dict[str, Any] | None) -> bool:
        """Check metadata equality or membership filter."""
        if not filters:
            return True
        from signalrag.retrieval.filter import MetadataFilter

        return MetadataFilter(filters).matches(chunk)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Query BM25 index and return ranked SearchResults with normalized scores."""
        query_text = query.strip()
        if not query_text or not self._bm25 or not self.chunks or top_k <= 0:
            return []

        tokens = default_tokenizer(query_text)
        if not tokens:
            return []

        raw_scores = self._bm25.get_scores(tokens)
        max_score = float(np.max(raw_scores)) if len(raw_scores) > 0 else 0.0

        if max_score <= 0.0:
            return []

        # Min-max normalization: scale scores to [0.0, 1.0] for hybrid compatibility
        normalized_scores = raw_scores / max_score

        # Sort candidate indices descending by score
        sorted_indices = np.argsort(-normalized_scores)

        results: list[SearchResult] = []
        rank = 1
        for idx in sorted_indices:
            score = float(normalized_scores[idx])
            if score < self.min_score or score <= 0.0:
                continue

            chunk = self.chunks[idx]
            if not self._matches_filters(chunk, filters):
                continue

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    retrieval_method="bm25",
                    rank=rank,
                    metadata_filters_applied={k: str(v) for k, v in (filters or {}).items()},
                )
            )
            rank += 1
            if len(results) >= top_k:
                break

        return results
