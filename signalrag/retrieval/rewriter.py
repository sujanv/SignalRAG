"""Query rewriting subsystem optimizing conversational user queries for retrieval."""

import re
from abc import ABC, abstractmethod
from typing import Any


class BaseQueryRewriter(ABC):
    """Abstract interface for query transformation and expansion."""

    @abstractmethod
    def rewrite(self, query: str) -> list[str]:
        """Transform user query into one or more retrieval-optimized queries."""
        pass


class HeuristicQueryRewriter(BaseQueryRewriter):
    """Rule-based query rewriter that cleans conversational preambles, expands acronyms, and extracts keywords."""

    QUESTION_FILLERS = [
        r"^(can\s+you\s+(please\s+)?(tell\s+me|explain|show\s+me)\s+)",
        r"^(could\s+you\s+(please\s+)?(tell\s+me|explain)\s+)",
        r"^(what\s+is\s+the\s+difference\s+between\s+)",
        r"^(what\s+is|what\s+are|what\s+'s)\s+",
        r"^(how\s+does|how\s+do|how\s+to|how\s+can\s+i)\s+",
        r"^(why\s+does|why\s+is|why\s+are)\s+",
        r"^(tell\s+me\s+about\s+)",
        r"^(i\s+want\s+to\s+know\s+about\s+)",
        r"^(please\s+explain\s+)",
    ]

    COMMON_ACRONYMS = {
        "rag": "retrieval augmented generation",
        "llm": "large language model",
        "llms": "large language models",
        "mrr": "mean reciprocal rank",
        "ndcg": "normalized discounted cumulative gain",
        "bm25": "best matching 25 lexical search",
        "nlp": "natural language processing",
        "api": "application programming interface",
    }

    def __init__(self, expand_acronyms: bool = True, strip_fillers: bool = True) -> None:
        self.expand_acronyms = expand_acronyms
        self.strip_fillers = strip_fillers

    def _strip_question_fillers(self, text: str) -> str:
        """Strip conversational filler patterns from query start."""
        cleaned = text.strip()
        for pattern in self.QUESTION_FILLERS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        # Clean trailing punctuation
        cleaned = re.sub(r"[\?\.\!]+$", "", cleaned).strip()
        return cleaned or text.strip()

    def _expand_acronyms(self, text: str) -> str | None:
        """Create an expanded query variant if acronyms are present."""
        words = text.split()
        expanded_words = []
        has_acronym = False

        for w in words:
            clean_w = w.lower().strip(".,;:?!()")
            if clean_w in self.COMMON_ACRONYMS:
                has_acronym = True
                expansion = self.COMMON_ACRONYMS[clean_w]
                expanded_words.append(f"{w} {expansion}")
            else:
                expanded_words.append(w)

        if has_acronym:
            return " ".join(expanded_words)
        return None

    def rewrite(self, query: str) -> list[str]:
        """Generate deduplicated list of candidate search queries."""
        raw = query.strip()
        if not raw:
            return []

        candidates: list[str] = []

        # 1. Cleaned core query
        cleaned = self._strip_question_fillers(raw) if self.strip_fillers else raw
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

        # 2. Acronym expanded variant
        if self.expand_acronyms:
            expanded = self._expand_acronyms(cleaned)
            if expanded and expanded not in candidates:
                candidates.append(expanded)

        # 3. Preserve original query as fallback if different
        if raw not in candidates:
            candidates.append(raw)

        return candidates


class LLMQueryRewriter(BaseQueryRewriter):
    """LLM-guided query rewriter producing alternative query formulations and HyDE passages."""

    def __init__(self, llm_client: Any = None, num_queries: int = 3) -> None:
        self.llm_client = llm_client
        self.num_queries = num_queries
        self._fallback = HeuristicQueryRewriter()

    def rewrite(self, query: str) -> list[str]:
        """Use LLM or fallback to heuristic query expansion."""
        if not self.llm_client:
            return self._fallback.rewrite(query)

        # If LLM client is configured, call prompt
        try:
            prompt = (
                f"You are a retrieval optimization assistant. "
                f"Given the user question, generate {self.num_queries} diverse search queries "
                f"optimized for semantic vector search and keyword retrieval.\n"
                f"Question: {query}\n"
                f"Return one query per line without numbers or bullets."
            )
            response = self.llm_client.generate(prompt)
            lines = [line.strip() for line in response.splitlines() if line.strip()]
            queries = [q for q in lines if not q.startswith(("#", "-", "*"))]
            if query not in queries:
                queries.append(query)
            return queries[: self.num_queries + 1]
        except Exception:
            return self._fallback.rewrite(query)
