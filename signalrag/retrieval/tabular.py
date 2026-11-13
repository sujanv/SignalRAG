"""Structured tabular data extraction, serialization, and retrieval for RAG pipelines."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class TableRow(BaseModel):
    """Single row in a structured table."""

    row_index: int
    cells: dict[str, str]


class TableChunk(BaseModel):
    """Parsed table representation preserving headers, rows, and serialized passage strings."""

    table_id: str
    document_id: str
    headers: list[str]
    rows: list[TableRow] = Field(default_factory=list)
    serialized_rows: list[str] = Field(default_factory=list)
    title: str = ""


class TabularRetriever:
    """Extracts markdown tables from documents, serializes them row-by-row with header semantics, and queries rows."""

    def extract_tables(self, text: str, document_id: str = "doc") -> list[TableChunk]:
        """Parse markdown formatted tables from text into structured TableChunk objects."""
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        tables: list[TableChunk] = []

        table_idx = 0
        i = 0
        while i < len(lines):
            # Look for table header line with pipe delimiters
            if lines[i].startswith("|") and lines[i].endswith("|") and i + 1 < len(lines):
                sep_line = lines[i + 1]
                if re.match(r"^\|[\s:-]+\|", sep_line):
                    # Valid markdown table found
                    headers = [c.strip() for c in lines[i].strip("|").split("|")]
                    rows: list[TableRow] = []
                    serialized: list[str] = []

                    j = i + 2
                    row_counter = 0
                    while j < len(lines) and lines[j].startswith("|") and lines[j].endswith("|"):
                        raw_cells = [c.strip() for c in lines[j].strip("|").split("|")]
                        cell_dict = {}
                        for h_idx, h in enumerate(headers):
                            val = raw_cells[h_idx] if h_idx < len(raw_cells) else ""
                            cell_dict[h] = val

                        row_obj = TableRow(row_index=row_counter, cells=cell_dict)
                        rows.append(row_obj)

                        # Row serialization with full column schema context
                        row_desc = ", ".join(f"{k}: {v}" for k, v in cell_dict.items() if v)
                        serialized.append(
                            f"[Table: {document_id}_tbl_{table_idx} | Row {row_counter + 1}] {row_desc}"
                        )

                        row_counter += 1
                        j += 1

                    tbl = TableChunk(
                        table_id=f"{document_id}_tbl_{table_idx}",
                        document_id=document_id,
                        headers=headers,
                        rows=rows,
                        serialized_rows=serialized,
                        title=f"Table {table_idx + 1} from {document_id}",
                    )
                    tables.append(tbl)
                    table_idx += 1
                    i = j
                    continue
            i += 1

        return tables

    def query_table(self, query: str, table: TableChunk) -> list[str]:
        """Search table rows by keyword/token overlap against row serialized text."""
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored_rows = []

        for row_str in table.serialized_rows:
            r_tokens = set(re.findall(r"\w+", row_str.lower()))
            overlap = len(q_tokens & r_tokens)
            if overlap > 0:
                scored_rows.append((overlap, row_str))

        scored_rows.sort(key=lambda x: x[0], reverse=True)
        if scored_rows:
            best_score = scored_rows[0][0]
            return [r for score, r in scored_rows if score == best_score]

        return table.serialized_rows[:2]
