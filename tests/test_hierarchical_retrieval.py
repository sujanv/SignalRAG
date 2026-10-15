"""Tests for Hierarchical Parent-Document retrieval."""

from signalrag.models.retrieval import SearchResult
from signalrag.retrieval.hierarchical import HierarchicalRetriever


def test_hierarchical_chunking_and_expansion():
    retriever = HierarchicalRetriever(parent_size=200, child_size=50)
    sample_text = (
        "Paragraph one explaining high-throughput vector similarity calculations. "
        "Paragraph two detailing BM25 inverted index term scoring algorithms. "
        "Paragraph three describing cross-encoder reciprocal rank fusion methods."
    )

    parents, children = retriever.build_hierarchy("doc1", sample_text, source="sample.txt")

    assert len(parents) >= 1
    assert len(children) > len(parents)

    # Test expansion of a child hit back to its parent
    child_hit = SearchResult(chunk=children[0], score=0.92)
    expanded = retriever.expand_results([child_hit])

    assert len(expanded) == 1
    assert expanded[0].chunk.id == parents[0].id
    assert len(expanded[0].chunk.text) >= len(children[0].text)
