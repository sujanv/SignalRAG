"""Unified SignalRAG engine orchestrating the complete RAG lifecycle."""

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from signalrag.core.config import Settings, get_settings
from signalrag.generation.citations import CitationTracker, TrackedAnswer
from signalrag.generation.guardrails import GroundedAnswerGuardrail, GuardrailEvaluation
from signalrag.generation.llm import BaseLLMClient, create_llm_client
from signalrag.generation.prompts import format_rag_prompt
from signalrag.generation.rendering import CitationRenderer
from signalrag.generation.streaming import StreamEvent, StreamingRAGResponse
from signalrag.models.citation import Citation
from signalrag.models.retrieval import SearchResult
from signalrag.retrieval.pipeline import RetrievalPipeline, RetrievalTrace


@dataclass
class RAGResponse:
    """Complete product response from SignalRAG."""

    answer: str
    query: str
    citations: list[Citation] = field(default_factory=list)
    guardrail: GuardrailEvaluation | None = None
    retrieved_results: list[SearchResult] = field(default_factory=list)
    trace: RetrievalTrace | None = None
    rendered_markdown: str = ""
    duration_seconds: float = 0.0


class SignalRAGEngine:
    """High-level facade orchestrating retrieval, generation, citation tracking, and guardrails."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        llm_client: BaseLLMClient | None = None,
        citation_tracker: CitationTracker | None = None,
        guardrail: GroundedAnswerGuardrail | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self.retrieval_pipeline = retrieval_pipeline
        self.llm_client = llm_client or create_llm_client(cfg.llm)
        self.citation_tracker = citation_tracker or CitationTracker()
        self.guardrail = guardrail or GroundedAnswerGuardrail()

    def ask(
        self,
        question: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        rerank: bool = True,
        compress: bool = False,
    ) -> RAGResponse:
        """Execute end-to-end question answering synchronously."""
        start_time = time.perf_counter()

        # 1. Retrieval pipeline execution
        pipe_response = self.retrieval_pipeline.retrieve_with_trace(
            query=question,
            top_k=top_k,
            filters=filters,
            rerank=rerank,
            compress=compress,
        )
        results = pipe_response.results
        trace = pipe_response.trace

        # 2. Check context sufficiency before generation
        sufficiency = self.guardrail.check_context_sufficiency(question, results)
        if sufficiency < self.guardrail.min_context_score:
            eval_refusal = GuardrailEvaluation(
                status=self.guardrail.evaluate(question, results, TrackedAnswer(raw_text="", cleaned_text="")).status,
                groundedness_score=0.0,
                context_sufficiency_score=sufficiency,
                reason="Insufficient context retrieved to answer the question reliably.",
                sanitized_answer=self.guardrail.STANDARD_REFUSAL,
            )
            return RAGResponse(
                answer=self.guardrail.STANDARD_REFUSAL,
                query=question,
                guardrail=eval_refusal,
                retrieved_results=results,
                trace=trace,
                rendered_markdown=self.guardrail.STANDARD_REFUSAL,
                duration_seconds=round(time.perf_counter() - start_time, 4),
            )

        # 3. LLM generation
        sys_prompt, user_prompt = format_rag_prompt(question, results)
        raw_text = self.llm_client.generate(prompt=user_prompt, system_prompt=sys_prompt)

        # 4. Citation tracking
        tracked = self.citation_tracker.track(raw_text, results)

        # 5. Guardrail evaluation
        guard_eval = self.guardrail.evaluate(question, results, tracked)

        # 6. Render final output
        final_text = guard_eval.sanitized_answer
        tracked_final = self.citation_tracker.track(final_text, results)
        rendered_md = CitationRenderer.render_markdown(tracked_final)

        total_duration = time.perf_counter() - start_time
        return RAGResponse(
            answer=final_text,
            query=question,
            citations=tracked_final.citations,
            guardrail=guard_eval,
            retrieved_results=results,
            trace=trace,
            rendered_markdown=rendered_md,
            duration_seconds=round(total_duration, 4),
        )

    def stream_ask(
        self,
        question: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        rerank: bool = True,
        compress: bool = False,
    ) -> StreamingRAGResponse:
        """Stream generated response tokens, followed by citations and guardrail status."""
        def _generator() -> Iterator[StreamEvent]:
            # 1. Retrieve
            pipe_response = self.retrieval_pipeline.retrieve_with_trace(
                query=question,
                top_k=top_k,
                filters=filters,
                rerank=rerank,
                compress=compress,
            )
            results = pipe_response.results
            trace = pipe_response.trace

            yield StreamEvent(event_type="trace", data={"raw_query": trace.raw_query, "candidates": len(results)})

            # 2. Check sufficiency
            sufficiency = self.guardrail.check_context_sufficiency(question, results)
            if sufficiency < self.guardrail.min_context_score:
                yield StreamEvent(event_type="token", data=self.guardrail.STANDARD_REFUSAL)
                yield StreamEvent(
                    event_type="guardrail",
                    data={"status": "refused", "reason": "Insufficient context relevance."},
                )
                yield StreamEvent(
                    event_type="complete",
                    data={"answer": self.guardrail.STANDARD_REFUSAL, "citations": []},
                )
                return

            # 3. Stream tokens
            sys_prompt, user_prompt = format_rag_prompt(question, results)
            accumulated_tokens: list[str] = []

            for token in self.llm_client.stream_generate(prompt=user_prompt, system_prompt=sys_prompt):
                accumulated_tokens.append(token)
                yield StreamEvent(event_type="token", data=token)

            full_text = "".join(accumulated_tokens)

            # 4. Citations & Guardrails
            tracked = self.citation_tracker.track(full_text, results)
            guard_eval = self.guardrail.evaluate(question, results, tracked)

            yield StreamEvent(
                event_type="guardrail",
                data={"status": guard_eval.status.value, "groundedness": guard_eval.groundedness_score},
            )
            yield StreamEvent(
                event_type="citation",
                data=[c.model_dump(mode="json") for c in tracked.citations],
            )
            yield StreamEvent(
                event_type="complete",
                data={
                    "answer": full_text,
                    "citations": [c.model_dump(mode="json") for c in tracked.citations],
                    "status": guard_eval.status.value,
                },
            )

        return StreamingRAGResponse(_generator())
