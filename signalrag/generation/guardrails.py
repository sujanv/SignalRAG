"""Grounded-answer guardrails preventing hallucinations and enforcing evidence sufficiency."""

import re
from dataclasses import dataclass
from enum import StrEnum

from signalrag.generation.citations import TrackedAnswer
from signalrag.models.retrieval import SearchResult


class GuardrailStatus(StrEnum):
    """Evaluation status of grounded-answer guardrails."""

    PASSED = "passed"
    REFUSED = "refused"
    QUALIFIED = "qualified"


@dataclass
class GuardrailEvaluation:
    """Detailed guardrail check result."""

    status: GuardrailStatus
    groundedness_score: float
    context_sufficiency_score: float
    reason: str
    sanitized_answer: str


class GroundedAnswerGuardrail:
    """Guardrail inspecting retrieved context sufficiency and answer groundedness."""

    STANDARD_REFUSAL = "I cannot find sufficient evidence in the provided documents to answer this question."

    REFUSAL_PATTERNS = [
        r"cannot find sufficient evidence",
        r"do not have enough information",
        r"the provided documents do not (contain|mention|state)",
        r"there is no mention of",
        r"not mentioned in the (provided|reference) documents",
    ]

    def __init__(
        self,
        min_context_score: float = 0.15,
        min_groundedness_score: float = 0.40,
        enforce_citations: bool = True,
    ) -> None:
        self.min_context_score = min_context_score
        self.min_groundedness_score = min_groundedness_score
        self.enforce_citations = enforce_citations

    def check_context_sufficiency(self, query: str, results: list[SearchResult]) -> float:
        """Measure whether the retrieved candidate pool has sufficient relevance."""
        if not results:
            return 0.0

        max_score = max(r.score for r in results)
        query_words = set(re.findall(r"\w+", query.lower()))

        # Check keyword presence across chunks
        combined_corpus = " ".join(r.chunk.text.lower() for r in results)
        matched_words = sum(1 for w in query_words if len(w) > 2 and w in combined_corpus)
        word_overlap = matched_words / max(1, len([w for w in query_words if len(w) > 2]))

        # Combined sufficiency score
        sufficiency = (0.5 * max_score) + (0.5 * word_overlap)
        return round(sufficiency, 4)

    def evaluate(
        self,
        query: str,
        results: list[SearchResult],
        tracked_answer: TrackedAnswer,
    ) -> GuardrailEvaluation:
        """Evaluate both context sufficiency and answer groundedness."""
        raw_text = tracked_answer.raw_text.strip()
        sufficiency = self.check_context_sufficiency(query, results)

        # 1. Check for explicit refusal by LLM
        for pat in self.REFUSAL_PATTERNS:
            if re.search(pat, raw_text, re.IGNORECASE):
                return GuardrailEvaluation(
                    status=GuardrailStatus.REFUSED,
                    groundedness_score=1.0,
                    context_sufficiency_score=sufficiency,
                    reason="Model explicitly declined to answer due to insufficient evidence.",
                    sanitized_answer=self.STANDARD_REFUSAL,
                )

        # 2. Check context sufficiency threshold
        if sufficiency < self.min_context_score:
            return GuardrailEvaluation(
                status=GuardrailStatus.REFUSED,
                groundedness_score=0.0,
                context_sufficiency_score=sufficiency,
                reason="Retrieved documents lack sufficient relevance to answer the question reliably.",
                sanitized_answer=self.STANDARD_REFUSAL,
            )

        # 3. Check groundedness of claims against cited chunks
        claim_scores: list[float] = []
        for claim_cite in tracked_answer.claim_citations:
            if not claim_cite.citations:
                # Claim makes assertions with no supporting citation
                claim_scores.append(0.0 if self.enforce_citations else 0.5)
                continue

            # Measure overlap between claim and cited chunk texts
            claim_words = set(re.findall(r"\w+", claim_cite.claim.lower()))
            if not claim_words:
                claim_scores.append(1.0)
                continue

            chunk_texts = " ".join(c.quote or "" for c in claim_cite.citations).lower()
            matched = sum(1 for w in claim_words if len(w) > 2 and w in chunk_texts)
            overlap = matched / max(1, len([w for w in claim_words if len(w) > 2]))
            claim_scores.append(overlap)

        groundedness = round(sum(claim_scores) / max(1, len(claim_scores)), 4) if claim_scores else 0.5

        if groundedness < self.min_groundedness_score:
            # Partially grounded answer - qualify response
            qualified_text = f"Based on limited document evidence: {raw_text}"
            return GuardrailEvaluation(
                status=GuardrailStatus.QUALIFIED,
                groundedness_score=groundedness,
                context_sufficiency_score=sufficiency,
                reason="Answer exhibits partial grounding; qualified for safety.",
                sanitized_answer=qualified_text,
            )

        return GuardrailEvaluation(
            status=GuardrailStatus.PASSED,
            groundedness_score=groundedness,
            context_sufficiency_score=sufficiency,
            reason="Answer is fully grounded in the provided reference documents.",
            sanitized_answer=raw_text,
        )
