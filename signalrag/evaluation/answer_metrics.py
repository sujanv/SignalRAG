"""Generation evaluation metrics: Faithfulness, Answer Relevance, and Citation Precision."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from signalrag.models.citation import CitationSource


def _tokenize(text: str) -> set[str]:
    """Extract lowercase alphanumeric tokens."""
    return set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower()))


class ClaimVerification(BaseModel):
    """Result of verifying an individual statement against context."""

    claim: str
    is_grounded: bool
    overlap_score: float
    supporting_chunks: list[str] = Field(default_factory=list)


class AnswerEvaluationResult(BaseModel):
    """Comprehensive evaluation scores for a generated response."""

    faithfulness: float = Field(description="Fraction of claims supported by context [0.0 - 1.0]")
    answer_relevance: float = Field(
        description="Relevance of response to query and ground truth [0.0 - 1.0]"
    )
    citation_precision: float = Field(
        description="Fraction of cited sources containing evidence [0.0 - 1.0]"
    )
    claims_evaluated: int = 0
    grounded_claims: int = 0
    hallucination_score: float = 0.0
    claim_details: list[ClaimVerification] = Field(default_factory=list)


class FaithfulnessEvaluator:
    """Evaluates whether statements in an answer are grounded in context chunks."""

    def __init__(self, token_overlap_threshold: float = 0.4):
        self.threshold = token_overlap_threshold

    def evaluate(
        self, answer: str, context_chunks: Sequence[str]
    ) -> tuple[float, list[ClaimVerification]]:
        """Evaluate claim grounding against context chunks."""
        # Split into sentence-like propositions
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]
        if not sentences:
            return 1.0, []

        chunk_token_sets = [_tokenize(c) for c in context_chunks]
        verifications: list[ClaimVerification] = []
        grounded_count = 0

        for sentence in sentences:
            sentence_tokens = _tokenize(sentence)
            if not sentence_tokens:
                continue

            best_overlap = 0.0
            supporting = []
            for idx, c_tokens in enumerate(chunk_token_sets):
                if not c_tokens:
                    continue
                intersection = len(sentence_tokens & c_tokens)
                overlap = intersection / len(sentence_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                if overlap >= self.threshold:
                    supporting.append(f"chunk_{idx}")

            is_grounded = best_overlap >= self.threshold
            if is_grounded:
                grounded_count += 1

            verifications.append(
                ClaimVerification(
                    claim=sentence,
                    is_grounded=is_grounded,
                    overlap_score=round(best_overlap, 3),
                    supporting_chunks=supporting,
                )
            )

        faithfulness = grounded_count / max(len(verifications), 1)
        return round(faithfulness, 4), verifications


class AnswerRelevanceEvaluator:
    """Evaluates question-answer and ground-truth answer alignment."""

    def evaluate(
        self, question: str, generated_answer: str, ground_truth: str | None = None
    ) -> float:
        """Compute answer relevance score."""
        q_tokens = _tokenize(question)
        ans_tokens = _tokenize(generated_answer)

        if not ans_tokens:
            return 0.0

        # Question coverage: how much of the question's core concepts appear in the answer
        q_overlap = len(q_tokens & ans_tokens) / max(len(q_tokens), 1)

        # Ground truth alignment if present
        if ground_truth:
            gt_tokens = _tokenize(ground_truth)
            if gt_tokens:
                intersection = len(ans_tokens & gt_tokens)
                f1 = 2 * intersection / (len(ans_tokens) + len(gt_tokens))
                # Combined score: weighted average of question relevance and ground-truth F1
                score = 0.3 * q_overlap + 0.7 * f1
                return round(min(max(score, 0.0), 1.0), 4)

        return round(min(max(q_overlap, 0.0), 1.0), 4)


class CitationPrecisionEvaluator:
    """Evaluates whether citations genuinely point to source chunks that support the claim."""

    def evaluate(self, citations: Sequence[CitationSource], context_chunks: Sequence[str]) -> float:
        """Calculate precision of cited passages."""
        if not citations:
            return 1.0  # No invalid citations claimed

        valid_citations = 0
        all_context_tokens = set()
        for c in context_chunks:
            all_context_tokens.update(_tokenize(c))

        for cit in citations:
            snippet_tokens = _tokenize(cit.snippet)
            if not snippet_tokens:
                valid_citations += 1
                continue
            # Citation snippet must match context
            if len(snippet_tokens & all_context_tokens) / len(snippet_tokens) >= 0.5:
                valid_citations += 1

        return round(valid_citations / len(citations), 4)


def evaluate_answer(
    question: str,
    generated_answer: str,
    context_chunks: Sequence[str],
    ground_truth: str | None = None,
    citations: Sequence[CitationSource] | None = None,
) -> AnswerEvaluationResult:
    """Execute all answer generation evaluations."""
    faith_eval = FaithfulnessEvaluator()
    rel_eval = AnswerRelevanceEvaluator()
    cit_eval = CitationPrecisionEvaluator()

    faithfulness, claim_details = faith_eval.evaluate(generated_answer, context_chunks)
    relevance = rel_eval.evaluate(question, generated_answer, ground_truth)
    citation_prec = cit_eval.evaluate(citations or [], context_chunks)

    grounded = sum(1 for c in claim_details if c.is_grounded)
    total_claims = len(claim_details)

    return AnswerEvaluationResult(
        faithfulness=faithfulness,
        answer_relevance=relevance,
        citation_precision=citation_prec,
        claims_evaluated=total_claims,
        grounded_claims=grounded,
        hallucination_score=round(1.0 - faithfulness, 4),
        claim_details=claim_details,
    )
