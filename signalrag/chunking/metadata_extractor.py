"""Metadata extractor for chunks preserving section hierarchy, page numbers, and offsets."""

import re
from typing import Any

from signalrag.models.chunk import ChunkMetadata
from signalrag.models.document import Document


class ChunkMetadataExtractor:
    """Enriches chunks with contextual metadata such as sections, pages, and offsets."""

    HEADER_PATTERN = re.compile(r"^(#{1,6}\s+.+|[A-Z0-9\.\s]{3,50}:)$", re.MULTILINE)
    PAGE_PATTERN = re.compile(r"---\s*Page\s+(\d+)\s*---", re.IGNORECASE)

    def __init__(self, track_sections: bool = True, track_pages: bool = True) -> None:
        self.track_sections = track_sections
        self.track_pages = track_pages

    def find_active_section(
        self, document_text: str, start_char: int, end_char: int
    ) -> tuple[str | None, list[str]]:
        """Find the active section heading and section hierarchy preceding or within the chunk."""
        if not self.track_sections or end_char <= 0:
            return None, []

        relevant_text = document_text[:end_char]
        matches = list(self.HEADER_PATTERN.finditer(relevant_text))
        if not matches:
            return None, []

        last_match = matches[-1]
        raw_header = last_match.group(0).strip()
        section_title = raw_header.lstrip("#").rstrip(":").strip()

        hierarchy: list[str] = []
        for m in matches[-3:]:
            h_text = m.group(0).strip().lstrip("#").rstrip(":").strip()
            if h_text and h_text not in hierarchy:
                hierarchy.append(h_text)

        return section_title, hierarchy

    def find_active_page(
        self, document_text: str, start_char: int, end_char: int, fallback_page: int | None = None
    ) -> int | None:
        """Find the active page number up to end_char based on page boundary markers."""
        if fallback_page is not None:
            return fallback_page

        if not self.track_pages or end_char < 0:
            return 1

        relevant_text = document_text[:end_char]
        matches = list(self.PAGE_PATTERN.finditer(relevant_text))
        if matches:
            try:
                return int(matches[-1].group(1))
            except ValueError:
                pass

        return 1

    def extract_metadata(
        self,
        document: Document,
        chunk_index: int,
        start_char: int,
        end_char: int,
        token_count: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChunkMetadata:
        """Create a complete ChunkMetadata instance with document inheritance and spatial tracking."""
        extra_meta = dict(document.metadata.extra)
        if extra:
            extra_meta.update(extra)

        section_title, hierarchy = self.find_active_section(document.text, start_char, end_char)
        if hierarchy:
            extra_meta["section_hierarchy"] = hierarchy

        fallback_page = document.metadata.extra.get("page_number")
        page_num = self.find_active_page(
            document.text, start_char, end_char, fallback_page=fallback_page
        )

        # Inherit document fields
        extra_meta["file_name"] = document.metadata.file_name
        extra_meta["file_type"] = document.metadata.file_type
        if document.metadata.author:
            extra_meta["author"] = document.metadata.author
        if document.metadata.title and not section_title:
            section_title = document.metadata.title

        return ChunkMetadata(
            document_id=document.id,
            chunk_index=chunk_index,
            source=document.metadata.source,
            page_number=page_num,
            start_char=start_char,
            end_char=end_char,
            section_title=section_title,
            token_count=token_count,
            extra=extra_meta,
        )
