"""Answer generation, citations, guardrails, streaming, and unified SignalRAGEngine."""

from signalrag.generation.citations import (
    CitationTracker,
    ClaimCitation,
    TrackedAnswer,
)
from signalrag.generation.engine import RAGResponse, SignalRAGEngine
from signalrag.generation.generator import AnswerGenerator, GeneratedAnswer
from signalrag.generation.guardrails import (
    GroundedAnswerGuardrail,
    GuardrailEvaluation,
    GuardrailStatus,
)
from signalrag.generation.llm import (
    BaseLLMClient,
    MockLLMClient,
    OpenAILLMClient,
    create_llm_client,
)
from signalrag.generation.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    format_context_block,
    format_rag_prompt,
)
from signalrag.generation.rendering import CitationRenderer
from signalrag.generation.streaming import StreamEvent, StreamingRAGResponse

__all__ = [
    "BaseLLMClient",
    "MockLLMClient",
    "OpenAILLMClient",
    "create_llm_client",
    "DEFAULT_SYSTEM_PROMPT",
    "format_context_block",
    "format_rag_prompt",
    "AnswerGenerator",
    "GeneratedAnswer",
    "CitationTracker",
    "ClaimCitation",
    "TrackedAnswer",
    "CitationRenderer",
    "GuardrailStatus",
    "GuardrailEvaluation",
    "GroundedAnswerGuardrail",
    "StreamEvent",
    "StreamingRAGResponse",
    "SignalRAGEngine",
    "RAGResponse",
]
