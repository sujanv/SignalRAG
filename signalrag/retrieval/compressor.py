"""Context compression removing irrelevant sentences from retrieved chunks."""

import re
from abc import ABC, abstractmethod

from signalrag.embeddings.base import BaseEmbeddingService, cosine_similarity
from signalrag.embeddings.hash_embedding import DeterministicHashEmbeddingService
from signalrag.models.retrieval import SearchResult


def split_sentences(text: str) -> list[str]:
    """Split text into sentences preserving meaningful punctuation."""
    # Split on periods, exclamation marks, or question marks followed by space or newline
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences or [text.strip()]


class BaseContextCompressor(ABC):
    """Abstract interface for compressing retrieved context chunks."""

    @abstractmethod
    def compress(
        self,
        query: str,
        results: list[SearchResult],
        max_sentences: int = 3,
        min_similarity: float = 0.2,
    ) -> list[SearchResult]:
        """Compress chunk texts to retain only salient sentences."""
        pass


class SentenceRelevanceCompressor(BaseContextCompressor):
    """Compresses chunks by scoring sentences against query and retaining top-scoring evidence."""

    def __init__(
        self,
        embedding_service: BaseEmbeddingService | None = None,
        min_sentence_score: float = 0.15,
        max_sentences_per_chunk: int = 4,
    ) -> None:
        self.embedding_service = embedding_service or DeterministicHashEmbeddingService(
            dimension=64
        )
        self.min_sentence_score = min_sentence_score
        self.max_sentences_per_chunk = max_sentences_per_chunk

    def compress(
        self,
        query: str,
        results: list[SearchResult],
        max_sentences: int | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]:
        if not results or not query.strip():
            return results

        limit = max_sentences or self.max_sentences_per_chunk
        threshold = min_similarity if min_similarity is not None else self.min_sentence_score

        query_vec = self.embedding_service.embed_text(query)
        query_words = set(re.findall(r"\w+", query.lower()))

        compressed_results: list[SearchResult] = []

        for res in results:
            new_res = res.model_copy(deep=True)
            sentences = split_sentences(new_res.chunk.text)

            if len(sentences) <= 1:
                compressed_results.append(new_res)
                continue

            # Score each sentence
            sentence_scores: list[tuple[int, str, float]] = []
            for s_idx, sentence in enumerate(sentences):
                # 1. Lexical overlap ratio
                s_words = set(re.findall(r"\w+", sentence.lower()))
                overlap = len(query_words.intersection(s_words)) / max(1, len(query_words))

                # 2. Embedding cosine similarity
                s_vec = self.embedding_service.embed_text(sentence)
                sim = cosine_similarity(query_vec, s_vec)

                # Combined score
                combined = (0.5 * overlap) + (0.5 * sim)
                sentence_scores.append((s_idx, sentence, combined))

            # Filter and select top sentences
            relevant = [item for item in sentence_scores if item[2] >= threshold]
            if not relevant:
                # Fallback to single top sentence if all below threshold
                relevant = sorted(sentence_scores, key=lambda x: x[2], reverse=True)[:1]
            else:
                relevant = sorted(relevant, key=lambda x: x[2], reverse=True)[:limit]

            # Re-order selected sentences back to original chronological appearance
            relevant.sort(key=lambda x: x[0])
            compressed_text = " ".join(item[1] for item in relevant)

            orig_len = len(new_res.chunk.text)
            comp_len = len(compressed_text)
            ratio = round(comp_len / max(1, orig_len), 3)

            new_res.chunk.text = compressed_text
            new_res.chunk.metadata.extra["original_char_length"] = orig_len
            new_res.chunk.metadata.extra["compressed_char_length"] = comp_len
            new_res.chunk.metadata.extra["compression_ratio"] = ratio

            compressed_results.append(new_res)

        return compressed_results
