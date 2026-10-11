"""Hallucination detection and claim-level attribution verification for RAG."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from signalrag.models.chunk import Chunk


class ClaimVerification(BaseModel):
    """Evaluation result for an atomic proposition in generated answer."""

    claim: str
    supported: bool
    confidence: float
    best_matching_chunk_id: str | None = None
    evidence_snippet: str | None = None


class HallucinationReport(BaseModel):
    """Aggregate hallucination and attribution assessment."""

    total_claims: int
    supported_claims: int
    unsupported_claims: int
    faithfulness_ratio: float
    verdict: str  # PASS, WARNING, FAIL
    claims: list[ClaimVerification] = Field(default_factory=list)


class HallucinationDetector:
    """Verifies factual grounding of generated responses against source chunks."""

    def __init__(self, support_threshold: float = 0.5):
        self.support_threshold = support_threshold

    def extract_claims(self, answer: str) -> list[str]:
        """Split answer text into individual verifiable statements."""
        raw_sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        claims = []
        for s in raw_sentences:
            cleaned = s.strip()
            # Filter trivial pleasantries or questions
            if len(cleaned) > 15 and not cleaned.endswith("?"):
                claims.append(cleaned)
        return claims or [answer.strip()]

    def verify_answer(self, answer: str, source_chunks: list[Chunk]) -> HallucinationReport:
        """Evaluate each claim against source chunks to detect hallucinations."""
        claims = self.extract_claims(answer)
        if not claims:
            return HallucinationReport(
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                faithfulness_ratio=1.0,
                verdict="PASS",
                claims=[],
            )

        verifications: list[ClaimVerification] = []
        supported_count = 0

        for claim in claims:
            claim_words = set(re.findall(r"\w+", claim.lower()))
            # Remove high-frequency stopwords
            content_words = {
                w
                for w in claim_words
                if w
                not in {
                    "the",
                    "a",
                    "an",
                    "is",
                    "are",
                    "was",
                    "were",
                    "this",
                    "that",
                    "in",
                    "on",
                    "at",
                    "by",
                    "for",
                    "with",
                }
            }

            best_score = 0.0
            best_chunk_id: str | None = None
            best_snippet: str | None = None

            for chunk in source_chunks:
                chunk_words = set(re.findall(r"\w+", chunk.text.lower()))
                if not content_words:
                    score = 1.0
                else:
                    overlap = len(content_words.intersection(chunk_words))
                    score = overlap / len(content_words)

                if score > best_score:
                    best_score = score
                    best_chunk_id = chunk.id
                    best_snippet = chunk.text[:100]

            is_supported = best_score >= self.support_threshold
            if is_supported:
                supported_count += 1

            verifications.append(
                ClaimVerification(
                    claim=claim,
                    supported=is_supported,
                    confidence=round(best_score, 3),
                    best_matching_chunk_id=best_chunk_id,
                    evidence_snippet=best_snippet,
                )
            )

        ratio = supported_count / len(claims) if claims else 1.0
        if ratio >= 0.8:
            verdict = "PASS"
        elif ratio >= 0.5:
            verdict = "WARNING"
        else:
            verdict = "FAIL"

        return HallucinationReport(
            total_claims=len(claims),
            supported_claims=supported_count,
            unsupported_claims=len(claims) - supported_count,
            faithfulness_ratio=round(ratio, 3),
            verdict=verdict,
            claims=verifications,
        )
