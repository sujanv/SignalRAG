"""Answer generator synthesizing grounded responses from retrieved candidate chunks."""

import time
from dataclasses import dataclass, field

from signalrag.core.config import Settings, get_settings
from signalrag.generation.llm import BaseLLMClient, create_llm_client
from signalrag.generation.prompts import format_rag_prompt
from signalrag.models.retrieval import SearchResult


@dataclass
class GeneratedAnswer:
    """Represents a generated RAG answer along with context and metadata."""

    answer: str
    query: str
    results: list[SearchResult] = field(default_factory=list)
    model_name: str = "mock-gpt-4o"
    duration_seconds: float = 0.0


class AnswerGenerator:
    """Takes user queries and retrieved chunks, prompts the LLM, and produces grounded answers."""

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self.llm_client = llm_client or create_llm_client(cfg.llm)

    def generate(self, query: str, results: list[SearchResult]) -> GeneratedAnswer:
        """Generate complete answer text from query and retrieved results."""
        start_time = time.perf_counter()

        system_prompt, user_prompt = format_rag_prompt(query, results)
        raw_answer = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        duration = time.perf_counter() - start_time
        model_name = getattr(self.llm_client, "model_name", "llm")

        return GeneratedAnswer(
            answer=raw_answer.strip(),
            query=query,
            results=results,
            model_name=model_name,
            duration_seconds=round(duration, 4),
        )
