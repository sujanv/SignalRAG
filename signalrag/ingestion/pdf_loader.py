"""PDF document ingestion pipeline with page tracking and text normalization."""

import hashlib
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pypdf import PdfReader
from pypdf.errors import EmptyFileError, PdfReadError

from signalrag.ingestion.base import BaseFileLoader
from signalrag.models.document import Document, DocumentMetadata


def normalize_pdf_text(text: str) -> str:
    """Clean and normalize extracted PDF text.

    - Resolves hyphenated linebreaks (e.g. 'informa-\\ntion' -> 'information')
    - Replaces odd unicode whitespace and null characters
    - Normalizes consecutive newlines and spaces
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Fix dehyphenation across line breaks: word-\nword -> wordword
    text = re.sub(r"(\b\w+)-\n(\w+\b)", r"\1\2", text)

    # Normalize carriage returns and tabs
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse sequences of 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize horizontal whitespace (spaces, tabs) without killing paragraph breaks
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    normalized = "\n".join(lines).strip()

    return normalized


class PDFLoader(BaseFileLoader):
    """Parses PDF files into normalized Document instances with page and document metadata."""

    def __init__(
        self,
        file_path: str | Path,
        mode: Literal["document", "page"] = "document",
        clean_text: bool = True,
        password: str | None = None,
    ) -> None:
        super().__init__(file_path)
        self.mode = mode
        self.clean_text = clean_text
        self.password = password

    def _extract_pdf(self) -> tuple[dict[int, str], dict[str, str | None]]:
        """Extract pages and metadata dictionary using pypdf."""
        try:
            reader = PdfReader(str(self.file_path))
            if reader.is_encrypted:
                if self.password:
                    decrypt_res = reader.decrypt(self.password)
                    if decrypt_res == 0:
                        raise ValueError(f"Incorrect password for encrypted PDF: {self.file_path}")
                else:
                    raise PermissionError(f"PDF is password protected: {self.file_path}")

            pages_text: dict[int, str] = {}
            for idx, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                pages_text[idx + 1] = normalize_pdf_text(raw_text) if self.clean_text else raw_text

            info = reader.metadata or {}
            meta_dict = {
                "title": getattr(info, "title", None) or info.get("/Title"),
                "author": getattr(info, "author", None) or info.get("/Author"),
                "creator": getattr(info, "creator", None) or info.get("/Creator"),
                "producer": getattr(info, "producer", None) or info.get("/Producer"),
            }
            return pages_text, meta_dict

        except (PdfReadError, EmptyFileError) as exc:
            raise ValueError(f"Malformed or corrupted PDF file '{self.file_path}': {exc}") from exc

    def load(self) -> list[Document]:
        """Load documents according to configured mode ('document' or 'page')."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """Yield parsed documents lazily."""
        stat = self.file_path.stat()
        file_size = stat.st_size
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)

        raw_bytes = self.file_path.read_bytes()
        file_hash = hashlib.sha256(raw_bytes).hexdigest()

        pages_text, pdf_info = self._extract_pdf()
        total_pages = len(pages_text)

        if self.mode == "page":
            # Yield one Document per page
            for page_num, text in pages_text.items():
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                page_doc_id = hashlib.sha256(
                    f"{self.file_path}:{page_num}:{content_hash}".encode()
                ).hexdigest()[:16]

                metadata = DocumentMetadata(
                    source=str(self.file_path),
                    file_name=self.file_path.name,
                    file_type="pdf",
                    file_size_bytes=file_size,
                    content_hash=content_hash,
                    title=pdf_info.get("title") or self.file_path.stem,
                    author=pdf_info.get("author"),
                    total_pages=total_pages,
                    modified_at=modified_at,
                    extra={
                        "page_number": page_num,
                        "file_hash": file_hash,
                        "creator": pdf_info.get("creator"),
                    },
                )
                yield Document(id=page_doc_id, text=text, metadata=metadata)
        else:
            # Aggregate all pages into one document with page headers
            full_text_parts = []
            for page_num, text in pages_text.items():
                if text.strip():
                    full_text_parts.append(f"--- Page {page_num} ---\n{text}")

            combined_text = "\n\n".join(full_text_parts)
            doc_id = hashlib.sha256(f"{self.file_path}:{file_hash}".encode()).hexdigest()[:16]

            metadata = DocumentMetadata(
                source=str(self.file_path),
                file_name=self.file_path.name,
                file_type="pdf",
                file_size_bytes=file_size,
                content_hash=file_hash,
                title=pdf_info.get("title") or self.file_path.stem,
                author=pdf_info.get("author"),
                total_pages=total_pages,
                modified_at=modified_at,
                extra={
                    "page_count": total_pages,
                    "creator": pdf_info.get("creator"),
                    "page_lengths": {p: len(txt) for p, txt in pages_text.items()},
                },
            )
            yield Document(id=doc_id, text=combined_text, metadata=metadata)
