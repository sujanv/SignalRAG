"""Tests for Structured Tabular RAG."""

from signalrag.retrieval.tabular import TabularRetriever


def test_tabular_retriever_parsing_and_query():
    raw_markdown = """
Here is the latency benchmark table:

| Backend | Metric | Score | Unit |
| --- | --- | --- | --- |
| BM25 | Recall@5 | 87.4 | % |
| Hybrid | Recall@5 | 93.1 | % |
| Dense | Latency | 1.2 | s |
"""

    retriever = TabularRetriever()
    tables = retriever.extract_tables(raw_markdown, document_id="bench_report")

    assert len(tables) == 1
    tbl = tables[0]
    assert tbl.headers == ["Backend", "Metric", "Score", "Unit"]
    assert len(tbl.rows) == 3
    assert len(tbl.serialized_rows) == 3
    assert "Hybrid" in tbl.serialized_rows[1]
    assert "93.1" in tbl.serialized_rows[1]

    # Query matching
    hits = retriever.query_table("Hybrid Recall", tbl)
    assert len(hits) == 1
    assert "Hybrid" in hits[0]
