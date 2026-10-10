"""Synthetic dataset generation from document chunks for RAG benchmarking."""

from __future__ import annotations

import re

from signalrag.evaluation.dataset import EvalExample, EvaluationDataset
from signalrag.models.chunk import Chunk


class SyntheticDatasetGenerator:
    """Generates synthetic question-answer-context evaluation triplets from chunks."""

    QUESTION_TEMPLATES = [
        "What are the key details regarding {topic}?",
        "According to the documentation, how does {topic} operate?",
        "Explain the primary mechanism of {topic}.",
        "What information is provided about {topic}?",
    ]

    def __init__(self, seed: int = 42):
        self.seed = seed

    def extract_keywords(self, text: str, max_words: int = 4) -> str:
        """Heuristically extract significant keywords from chunk text."""
        # Clean and find noun-like words
        words = re.findall(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b|\b[a-z]{5,}\b", text)
        filtered = [
            w
            for w in words
            if w.lower()
            not in {
                "these",
                "their",
                "which",
                "about",
                "there",
                "where",
                "would",
                "could",
                "should",
            }
        ]
        if not filtered:
            return "the specified topic"
        return " ".join(filtered[:max_words])

    def generate_example_from_chunk(self, chunk: Chunk, index: int = 1) -> EvalExample:
        """Create a single evaluation example anchored to a chunk."""
        # Find first meaningful sentence
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", chunk.text) if len(s.strip()) > 20]
        first_sentence = sentences[0] if sentences else chunk.text[:100]
        topic = self.extract_keywords(first_sentence)

        template = self.QUESTION_TEMPLATES[(index - 1) % len(self.QUESTION_TEMPLATES)]
        question = template.format(topic=topic)

        # Ground truth answer is derived from the source sentence/chunk
        ground_truth = first_sentence if first_sentence.endswith(".") else f"{first_sentence}."

        doc_id = chunk.document_id or (chunk.metadata.source if chunk.metadata else "unknown_doc")

        return EvalExample(
            id=f"synth_{index:04d}",
            question=question,
            ground_truth_answer=ground_truth,
            expected_doc_ids=[doc_id],
            expected_chunk_ids=[chunk.id],
            metadata={
                "generated": True,
                "strategy": "heuristic_chunk_anchor",
                "source_chunk_len": len(chunk.text),
            },
        )

    def generate_from_chunks(
        self, chunks: list[Chunk], max_examples: int | None = None
    ) -> EvaluationDataset:
        """Generate an evaluation dataset from a collection of chunks."""
        examples: list[EvalExample] = []
        target_chunks = chunks[:max_examples] if max_examples else chunks

        for idx, chunk in enumerate(target_chunks, 1):
            if len(chunk.text.strip()) < 30:
                continue
            ex = self.generate_example_from_chunk(chunk, index=idx)
            examples.append(ex)

        return EvaluationDataset(
            name="synthetic-rag-benchmark",
            description="Synthetically generated evaluation benchmark from document corpus",
            examples=examples,
        )
