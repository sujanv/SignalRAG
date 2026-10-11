"""Semantic and rule-assisted query router for adaptive retrieval pipelines."""

from __future__ import annotations

import re

from pydantic import BaseModel

from signalrag.models.retrieval import SearchResult
from signalrag.retrieval.pipeline import RetrievalPipeline


class RouteDecision(BaseModel):
    """Routing choice with confidence and rationale."""

    query: str
    target_strategy: str  # "bm25", "semantic", "hybrid", "multihop"
    confidence: float
    reasoning: str


class QueryRouter:
    """Classifies queries to direct them to the most effective retrieval backend."""

    IDENTIFIER_REGEX = re.compile(r"\b[A-Z0-9_]{3,}\b|['\"][^'\"]+['\"]|\d{3,}")
    COMPARATIVE_REGEX = re.compile(
        r"\b(compare|difference between|versus|vs\.?|how does .* differ)\b", re.IGNORECASE
    )
    CONCEPTUAL_REGEX = re.compile(
        r"\b(why|how does|explain|overview|conceptual|meaning of|what is)\b", re.IGNORECASE
    )

    def route(self, query: str) -> RouteDecision:
        """Analyze query and determine appropriate retrieval strategy."""
        q = query.strip()

        # Check for comparative / multi-part
        if self.COMPARATIVE_REGEX.search(q) or (" and " in q.lower() and len(q.split()) > 10):
            return RouteDecision(
                query=query,
                target_strategy="multihop",
                confidence=0.92,
                reasoning="Query contains comparative or multi-part clauses best resolved via decomposition.",
            )

        # Conceptual query
        if self.CONCEPTUAL_REGEX.search(q):
            return RouteDecision(
                query=query,
                target_strategy="hybrid",
                confidence=0.95,
                reasoning="Standard conceptual query benefits from hybrid fusion of dense and lexical signals.",
            )

        # Check for exact code/identifier/quoted terms
        if self.IDENTIFIER_REGEX.search(q) and len(q.split()) <= 4:
            return RouteDecision(
                query=query,
                target_strategy="bm25",
                confidence=0.88,
                reasoning="Query contains specific keywords, acronyms, or identifiers suitable for lexical search.",
            )

        # Default fallback
        return RouteDecision(
            query=query,
            target_strategy="hybrid",
            confidence=0.85,
            reasoning="Default balanced hybrid retrieval strategy.",
        )

    def execute_routed_query(
        self,
        query: str,
        pipeline: RetrievalPipeline,
        top_k: int = 5,
    ) -> tuple[RouteDecision, list[SearchResult]]:
        """Determine routing and execute query against retrieval pipeline."""
        decision = self.route(query)

        if decision.target_strategy == "bm25" and pipeline.bm25_retriever:
            results = pipeline.bm25_retriever.retrieve(query, top_k=top_k)
        elif decision.target_strategy == "semantic" and pipeline.semantic_retriever:
            results = pipeline.semantic_retriever.retrieve(query, top_k=top_k)
        else:
            # Fall back to unified pipeline
            results = pipeline.retrieve(query, top_k=top_k)

        return decision, results
