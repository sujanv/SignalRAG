"""LLM provider abstraction with streaming and mock client implementations."""

import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from signalrag.core.config import LLMConfig


class BaseLLMClient(ABC):
    """Abstract interface for LLM completion and generation."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generate response text for prompt."""
        pass

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Stream response tokens for prompt."""
        pass


class MockLLMClient(BaseLLMClient):
    """Deterministic offline mock LLM client synthesizing answers from reference documents."""

    def __init__(self, model_name: str = "mock-gpt-4o") -> None:
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        tokens = list(self.stream_generate(prompt, system_prompt, temperature, max_tokens))
        return "".join(tokens)

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        # Extract question
        q_match = re.search(r"Question:\s*(.*?)(?:\n\s*Answer:|$)", prompt, re.DOTALL)
        question = q_match.group(1).strip() if q_match else ""

        # Extract reference blocks: [1] Source: ... \n <text>
        doc_matches = list(re.finditer(r"\[(\d+)\]\s*Source:[^\n]*\n(.*?)(?=(?:\[\d+\]\s*Source:)|----------------|$)", prompt, re.DOTALL))

        if not doc_matches or not question:
            yield "I cannot find sufficient evidence in the provided documents to answer this question."
            return

        # Find matching documents
        q_words = set(re.findall(r"\w+", question.lower()))
        best_doc_idx = "1"
        best_sentence = ""
        max_overlap = -1

        for dm in doc_matches:
            d_idx = dm.group(1)
            d_text = dm.group(2).strip()
            sentences = re.split(r"(?<=[.!?])\s+", d_text)
            for s in sentences:
                s_words = set(re.findall(r"\w+", s.lower()))
                overlap = len(q_words.intersection(s_words))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_doc_idx = d_idx
                    best_sentence = s.strip()

        if max_overlap <= 0:
            yield "I cannot find sufficient evidence in the provided documents to answer this question."
            return

        # Synthesize grounded response with inline citation
        answer_parts = [
            f"Based on the provided documentation, {best_sentence.rstrip('.')} [{best_doc_idx}].",
        ]
        full_text = " ".join(answer_parts)

        # Stream words as tokens
        words = full_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")


class OpenAILLMClient(BaseLLMClient):
    """Live LLM client compatible with OpenAI and OpenAI-compatible endpoints."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(
                    api_key=self.api_key or "mock-key",
                    base_url=self.base_url,
                )
            except ImportError as err:
                raise ImportError("openai is not installed. Run `pip install openai`.") from err
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def create_llm_client(config: LLMConfig | None = None) -> BaseLLMClient:
    """Create LLM client based on configuration."""
    cfg = config or LLMConfig()
    provider = cfg.provider.lower()

    if provider in ("mock", "local"):
        return MockLLMClient(model_name=cfg.model)
    elif provider in ("openai", "ollama", "gemini"):
        # If no API key is provided and using OpenAI default, check env or use mock fallback
        api_key = cfg.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key and provider == "openai":
            return MockLLMClient(model_name=cfg.model)
        return OpenAILLMClient(
            model_name=cfg.model,
            api_key=api_key,
            base_url=cfg.base_url,
        )
    return MockLLMClient(model_name=cfg.model)
