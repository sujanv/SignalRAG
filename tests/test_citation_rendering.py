"""Tests for citation rendering in Markdown, plain text, and HTML formats."""

from signalrag.generation import CitationRenderer, CitationTracker
from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult


def test_citation_renderer_markdown():
    tracker = CitationTracker()
    c1 = Chunk.create(
        "d1",
        0,
        "Transformers utilize self-attention mechanisms to calculate pairwise token relevance.",
        "papers/attention.pdf",
        page_number=3,
        section_title="Attention Mechanism",
    )
    res = SearchResult(chunk=c1, score=0.95, rank=1)

    tracked = tracker.track("Transformers use pairwise self-attention [1].", [res])
    md_output = CitationRenderer.render_markdown(tracked)

    assert "Transformers use pairwise self-attention [1]." in md_output
    assert "### Sources" in md_output
    assert "[1] papers/attention.pdf — p. 3 (Attention Mechanism)" in md_output
    assert '> "Transformers utilize self-attention' in md_output


def test_citation_renderer_plain_text():
    tracker = CitationTracker()
    c1 = Chunk.create("d1", 0, "Fact A.", "doc.txt", page_number=2)
    res = SearchResult(chunk=c1, score=0.9, rank=1)

    tracked = tracker.track("Statement of fact [1].", [res])
    plain_output = CitationRenderer.render_plain_text(tracked)

    assert "Statement of fact [1]." in plain_output
    assert "Sources:" in plain_output
    assert "[1] doc.txt, p. 2" in plain_output


def test_citation_renderer_html():
    tracker = CitationTracker()
    c1 = Chunk.create("d1", 0, "Fact content.", "report.pdf", page_number=7)
    res = SearchResult(chunk=c1, score=0.88, rank=1)

    tracked = tracker.track("Document evidence demonstrates growth [1].", [res])
    html_output = CitationRenderer.render_html(tracked)

    assert '<a href="#cite-1">[1]</a>' in html_output
    assert '<li id="cite-1">' in html_output
    assert "report.pdf, p. 7" in html_output
    assert '<ol class="signalrag-citations">' in html_output
