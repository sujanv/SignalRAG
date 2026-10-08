"""Loader for plain text and markdown documents."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from signalrag.ingestion.base import BaseFileLoader
from signalrag.models.document import Document, DocumentMetadata


class TextLoader(BaseFileLoader):
    """Loads plain text, markdown, or code files from local filesystem."""

    def __init__(self, file_path: str | Path, encoding: str = "utf-8", autodetect_title: bool = True) -> None:
        super().__init__(file_path)
        self.encoding = encoding
        self.autodetect_title = autodetect_title

    def load(self) -> list[Document]:
        """Read and return document with extracted file metadata."""
        try:
            content = self.file_path.read_text(encoding=self.encoding)
        except UnicodeDecodeError:
            # Fallback to UTF-8 with replace to handle non-strict encodings safely
            content = self.file_path.read_text(encoding=self.encoding, errors="replace")

        stat = self.file_path.stat()
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        file_ext = self.file_path.suffix.lstrip(".").lower() or "text"

        title = None
        if self.autodetect_title and content.strip():
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            if lines:
                first_line = lines[0]
                if first_line.startswith("#"):
                    title = first_line.lstrip("#").strip()
                elif len(first_line) <= 120:
                    title = first_line

        metadata = DocumentMetadata(
            source=str(self.file_path),
            file_name=self.file_path.name,
            file_type=file_ext,
            file_size_bytes=stat.st_size,
            content_hash=content_hash,
            title=title,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            total_pages=1,
            extra={
                "extension": file_ext,
                "line_count": len(content.splitlines()),
            },
        )

        doc_id = hashlib.sha256(f"{self.file_path}:{content_hash}".encode()).hexdigest()[:16]
        return [Document(id=doc_id, text=content, metadata=metadata)]
