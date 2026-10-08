"""Recursive hierarchical text splitter."""

from collections.abc import Callable
from typing import Any

from signalrag.chunking.base import BaseChunker


class RecursiveCharacterChunker(BaseChunker):
    """Splits text recursively by trying separators in hierarchical order."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
        separators: list[str] | None = None,
        length_function: Callable[[str], int] | None = None,
        keep_separator: bool = True,
        metadata_extractor: Any = None,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            length_function=length_function,
            metadata_extractor=metadata_extractor,
        )
        self.separators = separators or list(self.DEFAULT_SEPARATORS)
        self.keep_separator = keep_separator

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursive splitting routine."""
        final_chunks: list[str] = []

        # Find best separator present in text
        separator = separators[-1]
        new_separators: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Split using selected separator
        if separator:
            if self.keep_separator:
                # Keep separator attached to fragments
                splits = text.split(separator)
                splits = [s + separator for s in splits[:-1]] + ([splits[-1]] if splits[-1] else [])
            else:
                splits = text.split(separator)
        else:
            # Character-by-character split
            splits = list(text)

        # Merge splits into chunks within chunk_size
        good_splits: list[str] = []
        for s in splits:
            if not s:
                continue
            if self.length_function(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits)
                    final_chunks.extend(merged)
                    good_splits = []
                if new_separators:
                    sub_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(s)

        if good_splits:
            merged = self._merge_splits(good_splits)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Combine fragments until chunk_size is reached, applying chunk_overlap."""
        docs: list[str] = []
        current_doc: list[str] = []
        total = 0

        for piece in splits:
            piece_len = self.length_function(piece)
            if total + piece_len > self.chunk_size and current_doc:
                doc_str = "".join(current_doc)
                if doc_str.strip():
                    docs.append(doc_str)

                # Keep overlap pieces from the tail
                while current_doc and total > self.chunk_overlap:
                    removed = current_doc.pop(0)
                    total -= self.length_function(removed)

            current_doc.append(piece)
            total += piece_len

        if current_doc:
            doc_str = "".join(current_doc)
            if doc_str.strip():
                docs.append(doc_str)

        return docs

    def chunk_text(self, text: str) -> list[str]:
        """Split text recursively into chunks."""
        if not text or not text.strip():
            return []
        return self._split_text(text, self.separators)
