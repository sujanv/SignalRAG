"""Base abstractions for document chunking."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import tiktoken

from signalrag.models.chunk import Chunk
from signalrag.models.document import Document


def get_token_counter(encoding_name: str = "cl100k_base") -> Callable[[str], int]:
    """Return a token counting function using tiktoken."""
    try:
        enc = tiktoken.get_encoding(encoding_name)
        return lambda text: len(enc.encode(text, disallowed_special=()))
    except Exception:
        # Fallback approximate token counter (~4 chars per token)
        return lambda text: max(1, len(text.split()))


class BaseChunker(ABC):
    """Abstract base class for chunking documents into smaller text fragments."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
        length_function: Callable[[str], int] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.length_function = length_function or len

    @abstractmethod
    def chunk_text(self, text: str) -> list[str]:
        """Split raw text into text segments."""
        pass

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Convert a Document into a list of Chunk instances preserving traceability."""
        raw_chunks = self.chunk_text(document.text)
        token_counter = get_token_counter()
        chunks: list[Chunk] = []

        # Find character offsets in document.text
        current_offset = 0
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text_stripped = chunk_text.strip()
            if not chunk_text_stripped or len(chunk_text_stripped) < self.min_chunk_size:
                continue

            start_idx = document.text.find(chunk_text_stripped, current_offset)
            if start_idx == -1:
                # If overlap shifted or modified formatting, search from 0
                start_idx = document.text.find(chunk_text_stripped)
                if start_idx == -1:
                    start_idx = current_offset

            end_idx = start_idx + len(chunk_text_stripped)
            current_offset = max(current_offset, start_idx + 1)

            token_count = token_counter(chunk_text_stripped)
            chunk = Chunk.create(
                document_id=document.id,
                chunk_index=idx,
                text=chunk_text_stripped,
                source=document.metadata.source,
                start_char=start_idx,
                end_char=end_idx,
                token_count=token_count,
                title=document.metadata.title,
            )
            chunks.append(chunk)

        return chunks
