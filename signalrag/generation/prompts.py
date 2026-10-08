"""Prompt templates and context formatting for grounded RAG generation."""

from signalrag.models.retrieval import SearchResult

DEFAULT_SYSTEM_PROMPT = """You are SignalRAG, a precise, trustworthy question-answering assistant.
Answer the user's question relying strictly and exclusively on the provided Reference Documents.

Guidelines:
1. Every factual statement or claim MUST be attributed to its source document using inline bracketed citations like [1], [2].
2. If the reference documents do not contain sufficient evidence to answer the question, state:
   "I cannot find sufficient evidence in the provided documents to answer this question."
3. Do not make assumptions or incorporate external knowledge not present in the excerpts.
4. Keep the answer clear, structured, and concise.
"""


def format_context_block(results: list[SearchResult]) -> str:
    """Format candidate search results into a clean numbered reference document block."""
    if not results:
        return "No reference documents provided."

    doc_blocks = []
    for idx, res in enumerate(results, start=1):
        chunk = res.chunk
        meta = chunk.metadata
        page_info = f", Page {meta.page_number}" if meta.page_number else ""
        section_info = f" ({meta.section_title})" if meta.section_title else ""
        header = f"[{idx}] Source: {meta.source}{page_info}{section_info}"
        doc_blocks.append(f"{header}\n{chunk.text}")

    return "\n\n".join(doc_blocks)


def format_rag_prompt(query: str, results: list[SearchResult]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the LLM."""
    context_str = format_context_block(results)
    user_prompt = f"""Reference Documents:
----------------------------------------
{context_str}
----------------------------------------

Question: {query}

Answer:"""
    return DEFAULT_SYSTEM_PROMPT, user_prompt
