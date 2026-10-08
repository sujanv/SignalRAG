"""Citation tracking mapping generated claims directly back to source chunks."""

import re
from dataclasses import dataclass, field

from signalrag.models.citation import Citation
from signalrag.models.retrieval import SearchResult


@dataclass
class ClaimCitation:
    """Links a specific assertion sentence to its supporting citation and chunk."""

    claim: str
    citation_indices: list[int]
    citations: list[Citation] = field(default_factory=list)
    support_score: float = 1.0


@dataclass
class TrackedAnswer:
    """Grounded answer with verified citation links and claim provenance."""

    raw_text: str
    cleaned_text: str
    citations: list[Citation] = field(default_factory=list)
    claim_citations: list[ClaimCitation] = field(default_factory=list)
    coverage_ratio: float = 1.0  # fraction of claims with at least one citation


class CitationTracker:
    """Extracts inline citation references from generated text and maps them to source chunks."""

    CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

    def extract_citation_indices(self, text: str) -> list[int]:
        """Extract all unique citation index integers in chronological appearance."""
        indices: list[int] = []
        for match in self.CITATION_PATTERN.finditer(text):
            parts = match.group(1).split(",")
            for p in parts:
                p_clean = p.strip()
                if p_clean.isdigit():
                    val = int(p_clean)
                    if val not in indices:
                        indices.append(val)
        return indices

    def _find_best_supporting_quote(self, claim_text: str, chunk_text: str) -> str:
        """Find the sentence in chunk_text that most closely supports the claim."""
        chunk_sentences = re.split(r"(?<=[.!?])\s+", chunk_text.strip())
        claim_words = set(re.findall(r"\w+", claim_text.lower()))

        best_sentence = chunk_text[:120]
        max_overlap = -1

        for s in chunk_sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_words = set(re.findall(r"\w+", s_clean.lower()))
            overlap = len(claim_words.intersection(s_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = s_clean

        return best_sentence

    def track(self, answer_text: str, results: list[SearchResult]) -> TrackedAnswer:
        """Parse answer text, map citations to results, and bind claims to source chunks."""
        if not answer_text.strip():
            return TrackedAnswer(raw_text="", cleaned_text="")

        # Map 1-indexed document numbers to SearchResult
        doc_map = dict(enumerate(results, start=1))

        # Split answer into sentences
        raw_sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())

        claim_citations: list[ClaimCitation] = []
        all_citations_map: dict[int, Citation] = {}
        total_claims = 0
        cited_claims = 0

        for sentence in raw_sentences:
            s_clean = sentence.strip()
            if not s_clean:
                continue

            total_claims += 1
            # Extract citations in this sentence
            indices_in_sentence = []
            for match in self.CITATION_PATTERN.finditer(s_clean):
                for p in match.group(1).split(","):
                    p_num = p.strip()
                    if p_num.isdigit():
                        idx_val = int(p_num)
                        if idx_val not in indices_in_sentence:
                            indices_in_sentence.append(idx_val)

            # Strip citation markers from the clean claim text
            clean_claim = self.CITATION_PATTERN.sub("", s_clean).strip()
            clean_claim = re.sub(r"\s+", " ", clean_claim)

            sentence_citations: list[Citation] = []
            if indices_in_sentence:
                cited_claims += 1
                for idx_val in indices_in_sentence:
                    if idx_val in doc_map:
                        res = doc_map[idx_val]
                        chunk = res.chunk
                        quote = self._find_best_supporting_quote(clean_claim, chunk.text)

                        citation = Citation(
                            index=idx_val,
                            source=chunk.metadata.source,
                            document_id=chunk.document_id,
                            chunk_id=chunk.id,
                            page_number=chunk.metadata.page_number,
                            section_title=chunk.metadata.section_title,
                            quote=quote,
                            relevance_score=res.score,
                        )
                        sentence_citations.append(citation)
                        all_citations_map[idx_val] = citation

            claim_citations.append(
                ClaimCitation(
                    claim=clean_claim,
                    citation_indices=indices_in_sentence,
                    citations=sentence_citations,
                )
            )

        # Sorted unique citations
        sorted_citations = [all_citations_map[k] for k in sorted(all_citations_map.keys())]
        coverage = round(cited_claims / max(1, total_claims), 3)

        return TrackedAnswer(
            raw_text=answer_text,
            cleaned_text=self.CITATION_PATTERN.sub("", answer_text).strip(),
            citations=sorted_citations,
            claim_citations=claim_citations,
            coverage_ratio=coverage,
        )
