"""Token-aware text chunker using tiktoken encodings with resilient fallback."""

import re
from typing import Any

import tiktoken

from signalrag.chunking.base import BaseChunker


class _FallbackWhitespaceTokenizer:
    """Fallback tokenizer when BPE encoding files cannot be loaded."""

    def encode(self, text: str, **kwargs: Any) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text)

    def decode(self, tokens: list[str]) -> str:
        # Rejoin with space heuristics
        return " ".join(tokens)


class TokenChunker(BaseChunker):
    """Chunks text based directly on token windows with token overlap."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
        encoding_name: str = "cl100k_base",
        metadata_extractor: Any = None,
    ) -> None:
        self.tokenizer: Any
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.tokenizer = _FallbackWhitespaceTokenizer()

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            metadata_extractor=metadata_extractor,
            length_function=lambda t: (
                len(self.tokenizer.encode(t, disallowed_special=()))
                if hasattr(self.tokenizer, "encode")
                else len(t.split())
            ),
        )

    def chunk_text(self, text: str) -> list[str]:
        """Split text into token windows."""
        if not text or not text.strip():
            return []

        try:
            tokens = self.tokenizer.encode(text, disallowed_special=())
        except TypeError:
            tokens = self.tokenizer.encode(text)

        if not tokens:
            return []

        chunks: list[str] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(tokens), step):
            token_slice = tokens[i : i + self.chunk_size]
            decoded = self.tokenizer.decode(token_slice).strip()
            if decoded:
                chunks.append(decoded)
            if i + self.chunk_size >= len(tokens):
                break

        return chunks
