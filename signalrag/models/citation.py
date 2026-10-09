"""Citation schema for grounded responses."""

from pydantic import BaseModel


class Citation(BaseModel):
    """Citation linking a generated claim directly back to source documents and chunks."""

    index: int
    source: str
    document_id: str
    chunk_id: str
    page_number: int | None = None
    section_title: str | None = None
    quote: str = ""
    snippet: str = ""
    relevance_score: float | None = None

    def render_marker(self) -> str:
        """Render markdown citation badge e.g. [1]."""
        return f"[{self.index}]"

    def render_source_line(self) -> str:
        """Render source attribution line."""
        page_str = f", p. {self.page_number}" if self.page_number else ""
        section_str = f" ({self.section_title})" if self.section_title else ""
        return f"[{self.index}] {self.source}{page_str}{section_str}"


class CitationSource(BaseModel):
    """Source chunk reference evaluated for citation precision."""

    chunk_id: str
    document_id: str = ""
    snippet: str = ""
    quote: str = ""
    score: float = 1.0

    @property
    def text(self) -> str:
        return self.snippet or self.quote
