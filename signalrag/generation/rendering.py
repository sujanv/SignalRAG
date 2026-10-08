"""Citation rendering formats for Markdown, plain text, and HTML footnotes."""

from signalrag.generation.citations import TrackedAnswer
from signalrag.models.citation import Citation


class CitationRenderer:
    """Renders cited answers into formatted Markdown, plain text, and HTML."""

    @classmethod
    def render_citation_line(cls, citation: Citation, include_quote: bool = True) -> str:
        """Render a single formatted source line."""
        page_str = f" — p. {citation.page_number}" if citation.page_number else ""
        section_str = f" ({citation.section_title})" if citation.section_title else ""
        header = f"[{citation.index}] {citation.source}{page_str}{section_info if (section_info := section_str) else ''}"

        if include_quote and citation.quote:
            clean_quote = citation.quote.replace("\n", " ").strip()
            return f"{header}\n    > \"{clean_quote}\""
        return header

    @classmethod
    def render_markdown(
        cls,
        tracked_answer: TrackedAnswer,
        include_quotes: bool = True,
        sources_header: str = "### Sources",
    ) -> str:
        """Render markdown answer with source citations section at bottom."""
        body = tracked_answer.raw_text.strip()
        if not tracked_answer.citations:
            return body

        sources_lines = [sources_header]
        for citation in tracked_answer.citations:
            sources_lines.append(cls.render_citation_line(citation, include_quote=include_quotes))

        sources_block = "\n\n".join(sources_lines)
        return f"{body}\n\n---\n\n{sources_block}"

    @classmethod
    def render_plain_text(cls, tracked_answer: TrackedAnswer) -> str:
        """Render simple plain text answer with source references."""
        body = tracked_answer.raw_text.strip()
        if not tracked_answer.citations:
            return body

        lines = [body, "\nSources:"]
        for c in tracked_answer.citations:
            page_str = f", p. {c.page_number}" if c.page_number else ""
            lines.append(f"[{c.index}] {c.source}{page_str}")
        return "\n".join(lines)

    @classmethod
    def render_html(cls, tracked_answer: TrackedAnswer) -> str:
        """Render HTML with hyperlinked footnote anchors."""
        import html
        import re

        text = html.escape(tracked_answer.raw_text)

        # Convert [1] to <sup><a href="#cite-1">[1]</a></sup>
        def _replace_anchor(match: re.Match) -> str:
            nums = [n.strip() for n in match.group(1).split(",")]
            anchors = [f'<a href="#cite-{n}">[{n}]</a>' for n in nums if n.isdigit()]
            return f"<sup>{', '.join(anchors)}</sup>"

        html_body = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", _replace_anchor, text)

        if not tracked_answer.citations:
            return f"<div class=\"signalrag-answer\">{html_body}</div>"

        cite_items = []
        for c in tracked_answer.citations:
            page_info = f", p. {c.page_number}" if c.page_number else ""
            sec_info = f" ({html.escape(c.section_title)})" if c.section_title else ""
            src = html.escape(c.source)
            quote_html = f'<blockquote class="citation-quote">"{html.escape(c.quote)}"</blockquote>' if c.quote else ""
            cite_items.append(
                f'<li id="cite-{c.index}">'
                f'<strong>[{c.index}]</strong> {src}{page_info}{sec_info}'
                f"{quote_html}</li>"
            )

        citations_html = '<ol class="signalrag-citations">\n' + "\n".join(cite_items) + "\n</ol>"
        return (
            f'<div class="signalrag-answer">\n'
            f'<div class="answer-content">{html_body}</div>\n'
            f'<hr class="signalrag-divider" />\n'
            f'<div class="answer-sources"><h4>Sources</h4>\n{citations_html}</div>\n'
            f"</div>"
        )
