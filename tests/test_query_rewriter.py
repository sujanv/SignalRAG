"""Tests for query rewriting and expansion."""

from signalrag.retrieval import HeuristicQueryRewriter, LLMQueryRewriter


def test_heuristic_query_rewriter_strips_fillers():
    rewriter = HeuristicQueryRewriter(expand_acronyms=False)

    queries = rewriter.rewrite("Can you please tell me how does hybrid search work?")
    assert len(queries) >= 1
    assert queries[0] == "how does hybrid search work" or "hybrid search work" in queries[0]


def test_heuristic_query_rewriter_acronym_expansion():
    rewriter = HeuristicQueryRewriter(expand_acronyms=True)

    queries = rewriter.rewrite("What is RAG evaluation?")
    assert len(queries) >= 2
    # Check that acronym expansion includes retrieval augmented generation
    assert any("retrieval augmented generation" in q.lower() for q in queries)


def test_llm_query_rewriter_fallback():
    # When no LLM client is configured, should safely fallback to heuristic rewriting
    rewriter = LLMQueryRewriter(llm_client=None)
    queries = rewriter.rewrite("Explain BM25 retrieval")
    assert len(queries) >= 1
    assert any("best matching 25" in q.lower() for q in queries)


def test_empty_query_rewrite():
    rewriter = HeuristicQueryRewriter()
    assert rewriter.rewrite("") == []
    assert rewriter.rewrite("   ") == []
