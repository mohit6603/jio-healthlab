"""The RAG pipeline.

    question -> embedding -> retrieval -> prompt -> generation -> answer + sources

Two behaviours are structural rather than left to the model:

* **No context, no answer.** When retrieval returns nothing above the score
  threshold, the pipeline returns a fixed "not enough information" message and
  never calls the model. A model asked to answer with an empty context block is
  being invited to hallucinate.
* **Sources are derived from retrieval, not parsed out of the answer.** The
  citation list is exactly what was retrieved, so it cannot drift from what the
  model was actually shown.
* **The healthcare boundary is enforced in code, not by the prompt.** Questions
  asking for a diagnosis, a personal result interpretation or treatment advice
  are answered with a fixed redirect and never reach the model; generated
  answers are re-checked before being returned. See ``rag/safety.py`` for why
  the system prompt alone was not sufficient.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from ..config import Settings, get_settings
from ..core.logging import get_logger
from ..llm.provider import LLMProvider
from ..schemas.rag import SearchHit
from .prompts import (
    AI_DISCLAIMER,
    INSUFFICIENT_CONTEXT_MESSAGE,
    SYSTEM_PROMPT,
    build_answer_prompt,
    build_report_explanation_prompt,
)
from .retriever import Retriever
from .safety import (
    CLINICAL_BOUNDARY_MESSAGE,
    contains_clinical_advice,
    requests_clinical_advice,
)

logger = get_logger(__name__)

#: Phrases that indicate the model declined for lack of grounding. Used only to
#: label the response, never to rewrite it.
_REFUSAL_MARKERS = (
    "not enough information",
    "don't have enough information",
    "do not have enough information",
    "does not contain",
    "doesn't contain",
    "no information",
    "cannot answer",
    "can't answer",
    "unable to answer",
)


@dataclass(slots=True)
class StreamEvent:
    """One event in a streamed answer.

    ``sources`` arrives first, then ``token`` events, then exactly one
    ``done``. A ``replace`` event means the safety screen rejected the
    completed text: the client must discard what it rendered.
    """

    type: str
    text: str = ""
    hits: list[SearchHit] = field(default_factory=list)
    grounded: bool = True
    finish_reason: str = "stop"
    model: str = ""
    provider: str = ""
    retrieval_ms: float = 0.0
    retrieval_count: int = 0
    total_ms: float = 0.0
    drop_sources: bool = False


@dataclass(slots=True)
class RagAnswer:
    """A grounded answer and everything needed to audit it."""

    question: str
    answer: str
    hits: list[SearchHit] = field(default_factory=list)
    #: False when the answer came from the fixed no-context path, or when the
    #: model declined for lack of grounding.
    grounded: bool = True
    disclaimer: str = AI_DISCLAIMER
    provider: str = "none"
    model: str = "none"
    finish_reason: str = "stop"
    prompt_truncated: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def retrieval_count(self) -> int:
        return len(self.hits)


class RagPipeline:
    """Composes retrieval and generation into a grounded answer."""

    def __init__(
        self,
        retriever: Retriever,
        provider: LLMProvider,
        settings: Settings | None = None,
    ) -> None:
        self._retriever = retriever
        self._provider = provider
        self._settings = settings or get_settings()

    # -------------------------------------------------------------- query ---
    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        category: str | None = None,
        max_new_tokens: int | None = None,
    ) -> RagAnswer:
        """Answer ``question`` from the knowledge base."""
        started = time.perf_counter()

        # Checked before retrieval and before generation: the model is never
        # given the chance to diagnose.
        if requests_clinical_advice(question):
            return self._boundary_answer(
                question, started, reason="clinical_boundary"
            )

        retrieval = self._retriever.retrieve(
            question,
            top_k=top_k,
            score_threshold=score_threshold,
            filters={"category": category} if category else None,
        )

        if retrieval.is_empty:
            return self._no_context_answer(retrieval.query, retrieval.total_ms, started)

        prompt = build_answer_prompt(retrieval.query, retrieval.hits)
        return self._generate(
            question=retrieval.query,
            prompt=prompt,
            hits=retrieval.hits,
            retrieval_ms=retrieval.total_ms,
            started=started,
            max_new_tokens=max_new_tokens,
        )

    # ----------------------------------------------------------- streaming ---
    def stream_answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        category: str | None = None,
        max_new_tokens: int | None = None,
    ) -> Iterator[StreamEvent]:
        """Answer incrementally, as a sequence of typed events.

        Sources are emitted *before* the first token, so the UI can show what
        the answer is grounded in while it is still being written. The refusal
        paths emit a single token event and stop -- they never reach the model,
        exactly as in the non-streaming path.
        """
        started = time.perf_counter()

        if requests_clinical_advice(question):
            logger.info("rag_clinical_boundary", extra={"reason": "stream"})
            yield StreamEvent("token", text=CLINICAL_BOUNDARY_MESSAGE)
            yield self._done_event(started, grounded=False, reason="clinical_boundary")
            return

        retrieval = self._retriever.retrieve(
            question,
            top_k=top_k,
            score_threshold=score_threshold,
            filters={"category": category} if category else None,
        )

        if retrieval.is_empty:
            logger.info("rag_no_context", extra={"generation_skipped": True})
            yield StreamEvent("token", text=INSUFFICIENT_CONTEXT_MESSAGE)
            yield self._done_event(started, grounded=False, reason="no_context")
            return

        yield StreamEvent("sources", hits=retrieval.hits)

        prompt = build_answer_prompt(retrieval.query, retrieval.hits)
        collected: list[str] = []

        for chunk in self._provider.stream(
            prompt, system=SYSTEM_PROMPT, max_new_tokens=max_new_tokens
        ):
            collected.append(chunk)
            yield StreamEvent("token", text=chunk)

        answer = "".join(collected).strip()

        # The safety screen can only run on the finished text. If it fires, the
        # UI is told to discard what it has rendered and show the redirect
        # instead -- which is why the event carries `replace`.
        if answer and contains_clinical_advice(answer):
            logger.warning("clinical_advice_suppressed", extra={"streamed": True})
            yield StreamEvent("replace", text=CLINICAL_BOUNDARY_MESSAGE)
            yield self._done_event(
                started, grounded=False, reason="clinical_boundary", drop_sources=True
            )
            return

        grounded = bool(answer) and not looks_like_refusal(answer)
        yield self._done_event(
            started,
            grounded=grounded,
            reason="stop",
            retrieval_ms=retrieval.total_ms,
            retrieval_count=len(retrieval.hits),
        )

    def _done_event(
        self,
        started: float,
        *,
        grounded: bool,
        reason: str,
        retrieval_ms: float = 0.0,
        retrieval_count: int = 0,
        drop_sources: bool = False,
    ) -> StreamEvent:
        return StreamEvent(
            "done",
            grounded=grounded,
            finish_reason=reason,
            model=self._provider.model_name,
            provider=self._provider.name,
            retrieval_ms=retrieval_ms,
            retrieval_count=retrieval_count,
            total_ms=round((time.perf_counter() - started) * 1000, 2),
            drop_sources=drop_sources,
        )

    # ------------------------------------------------------------- explain ---
    def explain_report(
        self,
        report_summary: str,
        *,
        search_text: str,
        top_k: int | None = None,
        max_new_tokens: int | None = None,
    ) -> RagAnswer:
        """Explain a sanitised laboratory request in general terms.

        ``report_summary`` must contain no patient identifiers; the backend
        sanitises before calling. ``search_text`` is what the knowledge base is
        searched with (typically the test name).
        """
        started = time.perf_counter()
        retrieval = self._retriever.retrieve(search_text, top_k=top_k)

        if retrieval.is_empty:
            return self._no_context_answer(search_text, retrieval.total_ms, started)

        prompt = build_report_explanation_prompt(report_summary, retrieval.hits)
        return self._generate(
            question=search_text,
            prompt=prompt,
            hits=retrieval.hits,
            retrieval_ms=retrieval.total_ms,
            started=started,
            max_new_tokens=max_new_tokens,
        )

    # ------------------------------------------------------------ internals --
    def _boundary_answer(
        self, question: str, started: float, *, reason: str
    ) -> RagAnswer:
        """Redirect to a clinician. Never carries model output or sources."""
        logger.info("rag_clinical_boundary", extra={"reason": reason})
        return RagAnswer(
            question=question,
            answer=CLINICAL_BOUNDARY_MESSAGE,
            hits=[],
            grounded=False,
            provider=self._provider.name,
            model=self._provider.model_name,
            finish_reason=reason,
            total_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def _no_context_answer(
        self, question: str, retrieval_ms: float, started: float
    ) -> RagAnswer:
        """Refuse honestly. The model is not consulted at all."""
        logger.info(
            "rag_no_context",
            extra={"retrieval_ms": retrieval_ms, "generation_skipped": True},
        )
        return RagAnswer(
            question=question,
            answer=INSUFFICIENT_CONTEXT_MESSAGE,
            hits=[],
            grounded=False,
            provider=self._provider.name,
            model=self._provider.model_name,
            finish_reason="no_context",
            retrieval_ms=retrieval_ms,
            generation_ms=0.0,
            total_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def _generate(
        self,
        *,
        question: str,
        prompt: str,
        hits: list[SearchHit],
        retrieval_ms: float,
        started: float,
        max_new_tokens: int | None,
    ) -> RagAnswer:
        result = self._provider.generate(
            prompt, system=SYSTEM_PROMPT, max_new_tokens=max_new_tokens
        )

        answer = result.text.strip()
        # An empty completion is not an answer; report it as ungrounded rather
        # than returning a blank bubble to the user.
        if not answer:
            answer = INSUFFICIENT_CONTEXT_MESSAGE
            grounded = False
        elif contains_clinical_advice(answer):
            # Defence in depth: the model crossed the boundary despite the
            # system prompt. Replace the answer rather than surface it.
            logger.warning(
                "clinical_advice_suppressed",
                extra={"model": result.model, "retrieval_count": len(hits)},
            )
            answer = CLINICAL_BOUNDARY_MESSAGE
            grounded = False
            hits = []
        else:
            grounded = not looks_like_refusal(answer)

        total_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "rag_answer_generated",
            extra={
                "retrieval_count": len(hits),
                "grounded": grounded,
                "provider": result.provider,
                "model": result.model,
                "finish_reason": result.finish_reason,
                "prompt_truncated": result.prompt_truncated,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "retrieval_ms": retrieval_ms,
                "generation_ms": result.latency_ms,
                "total_ms": total_ms,
            },
        )

        return RagAnswer(
            question=question,
            answer=answer,
            # Citations mirror what was retrieved, so they cannot disagree with
            # what the model was shown.
            hits=hits,
            grounded=grounded,
            provider=result.provider,
            model=result.model,
            finish_reason=result.finish_reason,
            prompt_truncated=result.prompt_truncated,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            retrieval_ms=retrieval_ms,
            generation_ms=result.latency_ms,
            total_ms=total_ms,
        )


def looks_like_refusal(answer: str) -> bool:
    """Whether the model declined for lack of grounding.

    Used only to set the ``grounded`` flag so the UI can present the answer
    honestly. The answer text itself is never rewritten on the basis of this.
    """
    lowered = answer.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)
