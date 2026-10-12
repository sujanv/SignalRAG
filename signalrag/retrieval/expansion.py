"""Pseudo-Relevance Feedback (PRF) and query expansion for vocabulary mismatch resolution."""

from __future__ import annotations

import collections
import re

from pydantic import BaseModel, Field

from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult
from signalrag.retrieval.pipeline import RetrievalPipeline


class ExpandedQuery(BaseModel):
    """Result of query expansion with attribution to discovered terms."""

    original_query: str
    expanded_query: str
    expansion_terms: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class PseudoRelevanceExpander:
    """Extracts salient vocabulary from top-k retrieved passages to enrich queries."""

    STOPWORDS = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "where",
        "when",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "it",
        "its",
        "of",
        "as",
    }

    def __init__(self, top_n_terms: int = 3, term_weight: float = 0.5):
        self.top_n_terms = top_n_terms
        self.term_weight = term_weight

    def extract_terms(self, text: str) -> list[str]:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return [w for w in words if w not in self.STOPWORDS]

    def expand(self, query: str, feedback_chunks: list[Chunk]) -> ExpandedQuery:
        """Derive expansion terms from feedback chunks and build enriched query string."""
        if not feedback_chunks:
            return ExpandedQuery(
                original_query=query,
                expanded_query=query,
                expansion_terms=[],
                confidence=1.0,
            )

        query_terms = set(self.extract_terms(query))
        term_counts: collections.Counter[str] = collections.Counter()

        for chunk in feedback_chunks:
            words = self.extract_terms(chunk.text)
            for w in words:
                if w not in query_terms:
                    term_counts[w] += 1

        selected_terms = [t for t, _ in term_counts.most_common(self.top_n_terms)]

        if not selected_terms:
            return ExpandedQuery(
                original_query=query,
                expanded_query=query,
                expansion_terms=[],
                confidence=1.0,
            )

        expanded_str = f"{query.strip()} {' '.join(selected_terms)}"
        return ExpandedQuery(
            original_query=query,
            expanded_query=expanded_str,
            expansion_terms=selected_terms,
            confidence=round(1.0 + (len(selected_terms) * 0.1), 2),
        )

    def retrieve_with_prf(
        self,
        query: str,
        pipeline: RetrievalPipeline,
        top_k: int = 5,
        feedback_k: int = 3,
    ) -> tuple[ExpandedQuery, list[SearchResult]]:
        """Run initial retrieval, derive expansion terms, and execute final retrieval pass."""
        initial_results = pipeline.retrieve(query, top_k=feedback_k)
        feedback_chunks = [r.chunk for r in initial_results]

        exp_query = self.expand(query, feedback_chunks)
        final_results = pipeline.retrieve(exp_query.expanded_query, top_k=top_k)

        return exp_query, final_results
