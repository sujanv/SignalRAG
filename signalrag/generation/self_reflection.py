"""Self-Reflective RAG (Self-RAG) grading retrieval relevance and triggering dynamic re-retrieval."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from signalrag.models.chunk import Chunk


class SelfReflectionDecision(BaseModel):
    """Critique assessment and adaptive execution path for generation."""

    retrieval_grade: str  # RELEVANT, PARTIAL, IRRELEVANT
    relevance_score: float
    needs_re_retrieval: bool
    reformulated_query: str | None = None
    action: str  # GENERATE, RE_RETRIEVE, REFUSE
    critique_notes: list[str] = Field(default_factory=list)


class SelfReflectiveEngine:
    """Evaluates passage relevance self-reflectively to prevent answering from poor context."""

    def __init__(self, relevance_threshold: float = 0.45):
        self.relevance_threshold = relevance_threshold

    def grade_retrieval(self, query: str, chunks: list[Chunk]) -> SelfReflectionDecision:
        """Inspect candidate passages and assign a self-reflection critique grade."""
        if not chunks:
            return SelfReflectionDecision(
                retrieval_grade="IRRELEVANT",
                relevance_score=0.0,
                needs_re_retrieval=True,
                reformulated_query=f"Overview of {query.strip()}",
                action="RE_RETRIEVE",
                critique_notes=["Empty retrieval candidate set."],
            )

        q_terms = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", query.lower()))
        if not q_terms:
            q_terms = {"general"}

        scores = []
        for c in chunks:
            c_words = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", c.text.lower()))
            overlap = len(q_terms.intersection(c_words))
            score = overlap / len(q_terms)
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        if avg_score >= self.relevance_threshold:
            grade = "RELEVANT"
            action = "GENERATE"
            re_ret = False
            re_q = None
            notes = ["Candidate passages contain sufficient semantic alignment with query terms."]
        elif avg_score >= 0.2:
            grade = "PARTIAL"
            action = "GENERATE"
            re_ret = False
            re_q = None
            notes = ["Passages provide partial alignment; answer with qualification."]
        else:
            grade = "IRRELEVANT"
            action = "RE_RETRIEVE"
            re_ret = True
            re_q = f"detailed background on {query.strip()}"
            notes = ["Low keyword and conceptual overlap detected; triggering query reformulation."]

        return SelfReflectionDecision(
            retrieval_grade=grade,
            relevance_score=round(avg_score, 3),
            needs_re_retrieval=re_ret,
            reformulated_query=re_q,
            action=action,
            critique_notes=notes,
        )
