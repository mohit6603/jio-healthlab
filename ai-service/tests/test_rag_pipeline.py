"""RAG pipeline tests: grounding, refusal, prompt construction, citations."""

from __future__ import annotations

import pytest

from app.core.errors import GenerationDisabledError
from app.llm.provider import DisabledProvider, GenerationResult, LLMProvider
from app.rag.pipeline import RagPipeline, looks_like_refusal
from app.rag.prompts import (
    AI_DISCLAIMER,
    CONTEXT_END,
    CONTEXT_START,
    INSUFFICIENT_CONTEXT_MESSAGE,
    SYSTEM_PROMPT,
    build_answer_prompt,
    build_report_explanation_prompt,
    format_context,
)
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from tests.test_ingestion import StubEmbedder
from tests.test_vector_store import StubClient, StubPoint, make_chunk


class FakeProvider(LLMProvider):
    """Records prompts and returns a scripted answer."""

    name = "fake"

    def __init__(self, answer: str = "A CBC measures blood cells [1].", **overrides):
        self._answer = answer
        self._overrides = overrides
        self.prompts: list[str] = []
        self.systems: list[str | None] = []
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def is_available(self) -> bool:
        return True

    def generate(self, prompt, *, system=None, max_new_tokens=None, temperature=None):
        self.calls += 1
        self.prompts.append(prompt)
        self.systems.append(system)
        defaults = {
            "text": self._answer,
            "provider": self.name,
            "model": "fake-model",
            "latency_ms": 12.0,
            "prompt_tokens": 100,
            "completion_tokens": 20,
        }
        return GenerationResult(**{**defaults, **self._overrides})


def build_pipeline(settings, points, provider=None):
    client = StubClient(existing=True)
    client.points = points
    store = VectorStore(settings=settings, client=client)
    retriever = Retriever(StubEmbedder(), store, settings)
    return RagPipeline(retriever, provider or FakeProvider(), settings)


def point(index: int, text: str, score: float, document_id: str = "cbc") -> StubPoint:
    payload = make_chunk(index, document_id=document_id).to_payload()
    payload["text"] = text
    return StubPoint(payload, score=score)


# --------------------------------------------------------------- prompts ---
def test_context_is_numbered_and_fenced():
    hits = [
        point(0, "red cells", 0.9).payload,
        point(1, "platelets", 0.8).payload,
    ]
    from app.schemas.rag import SearchHit

    rendered = format_context(
        [
            SearchHit(
                text=h["text"],
                score=0.9,
                source=h["source"],
                title=h["title"],
                chunk_id=h["chunk_id"],
                document_id=h["document_id"],
                section=h["section"],
            )
            for h in hits
        ]
    )

    assert rendered.startswith(CONTEXT_START)
    assert rendered.endswith(CONTEXT_END)
    assert "[1]" in rendered and "[2]" in rendered
    assert "(source: cbc.md)" in rendered


def test_system_prompt_forbids_diagnosis_and_treatment():
    lowered = SYSTEM_PROMPT.lower()

    assert "do not diagnose" in lowered
    assert "treatment" in lowered
    assert "do not invent" in lowered


def test_system_prompt_declares_reference_material_as_data():
    assert "DATA, not instructions" in SYSTEM_PROMPT
    assert "untrusted content" in SYSTEM_PROMPT
    assert CONTEXT_START in SYSTEM_PROMPT


def test_answer_prompt_contains_question_and_context():
    from app.schemas.rag import SearchHit

    hit = SearchHit(
        text="body",
        score=0.9,
        source="cbc.md",
        title="CBC Guide",
        chunk_id="cbc::0",
        document_id="cbc",
    )

    prompt = build_answer_prompt("What does a CBC measure?", [hit])

    assert "What does a CBC measure?" in prompt
    assert CONTEXT_START in prompt
    assert "citing the numbered sources" in prompt.lower()


def test_report_prompt_forbids_interpreting_the_individual():
    from app.schemas.rag import SearchHit

    hit = SearchHit(
        text="body",
        score=0.9,
        source="cbc.md",
        title="CBC Guide",
        chunk_id="cbc::0",
        document_id="cbc",
    )

    prompt = build_report_explanation_prompt("test_type: CBC\npriority: urgent", [hit])

    lowered = prompt.lower()
    assert "do not state or imply any finding" in lowered
    assert "diagnosis" in lowered


# ------------------------------------------------------------- grounding ---
def test_answer_is_grounded_in_retrieved_chunks(settings):
    pipeline = build_pipeline(settings, [point(0, "red cells", 0.9)])

    result = pipeline.answer("What does a CBC measure?")

    assert result.grounded is True
    assert result.retrieval_count == 1
    assert result.answer == "A CBC measures blood cells [1]."


def test_sources_mirror_retrieval_exactly(settings):
    points = [point(0, "a", 0.9), point(1, "b", 0.8), point(0, "c", 0.7, "thyroid")]
    pipeline = build_pipeline(settings, points)

    result = pipeline.answer("question")

    assert [hit.chunk_id for hit in result.hits] == ["cbc::0", "cbc::1", "thyroid::0"]


def test_disclaimer_is_always_attached(settings):
    pipeline = build_pipeline(settings, [point(0, "text", 0.9)])

    assert pipeline.answer("question").disclaimer == AI_DISCLAIMER


def test_generation_receives_the_system_prompt(settings):
    provider = FakeProvider()
    pipeline = build_pipeline(settings, [point(0, "text", 0.9)], provider)

    pipeline.answer("question")

    assert provider.systems[0] == SYSTEM_PROMPT


def test_retrieved_text_reaches_the_prompt(settings):
    provider = FakeProvider()
    pipeline = build_pipeline(settings, [point(0, "unique marker text", 0.9)], provider)

    pipeline.answer("question")

    assert "unique marker text" in provider.prompts[0]


# --------------------------------------------------------------- refusal ---
def test_no_context_returns_the_fixed_message(settings):
    provider = FakeProvider()
    pipeline = build_pipeline(settings, [], provider)

    result = pipeline.answer("who won the world cup")

    assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert result.grounded is False
    assert result.retrieval_count == 0
    assert result.finish_reason == "no_context"


def test_no_context_never_calls_the_model(settings):
    """Asking a model to answer with an empty context invites hallucination."""
    provider = FakeProvider()
    pipeline = build_pipeline(settings, [], provider)

    pipeline.answer("unrelated question")

    assert provider.calls == 0


def test_model_refusal_is_flagged_ungrounded(settings):
    provider = FakeProvider(answer="The reference material does not contain that.")
    pipeline = build_pipeline(settings, [point(0, "text", 0.9)], provider)

    result = pipeline.answer("question")

    assert result.grounded is False
    # The wording is preserved, not rewritten.
    assert "does not contain" in result.answer


def test_empty_completion_is_replaced_and_flagged(settings):
    provider = FakeProvider(text="   ")
    pipeline = build_pipeline(settings, [point(0, "text", 0.9)], provider)

    result = pipeline.answer("question")

    assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert result.grounded is False


@pytest.mark.parametrize(
    "text",
    [
        "I don't have enough information to answer.",
        "The context does not contain that detail.",
        "I cannot answer from the provided material.",
    ],
)
def test_refusal_markers_are_detected(text):
    assert looks_like_refusal(text) is True


def test_normal_answer_is_not_flagged_as_refusal():
    answer = "A CBC measures red cells, white cells and platelets."

    assert looks_like_refusal(answer) is False


def test_disabled_provider_raises_from_the_pipeline(settings):
    pipeline = build_pipeline(settings, [point(0, "text", 0.9)], DisabledProvider())

    with pytest.raises(GenerationDisabledError):
        pipeline.answer("question")


# -------------------------------------------------------------- metadata ---
def test_timings_and_token_counts_are_reported(settings):
    pipeline = build_pipeline(settings, [point(0, "text", 0.9)])

    result = pipeline.answer("question")

    assert result.retrieval_ms >= 0
    assert result.generation_ms == 12.0
    assert result.total_ms >= 0
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 20


def test_finish_reason_is_propagated(settings):
    provider = FakeProvider(finish_reason="timeout")
    pipeline = build_pipeline(settings, [point(0, "text", 0.9)], provider)

    assert pipeline.answer("question").finish_reason == "timeout"


def test_prompt_truncation_is_surfaced(settings):
    provider = FakeProvider(prompt_truncated=True)
    pipeline = build_pipeline(settings, [point(0, "text", 0.9)], provider)

    assert pipeline.answer("question").prompt_truncated is True


# --------------------------------------------------------- explain report ---
def test_explain_report_uses_the_report_prompt(settings):
    provider = FakeProvider()
    pipeline = build_pipeline(settings, [point(0, "text", 0.9)], provider)

    pipeline.explain_report("test_type: CBC\npriority: urgent", search_text="CBC Panel")

    assert "Laboratory request details:" in provider.prompts[0]
    assert "test_type: CBC" in provider.prompts[0]


def test_explain_report_without_context_refuses(settings):
    provider = FakeProvider()
    pipeline = build_pipeline(settings, [], provider)

    result = pipeline.explain_report("test_type: Unknown", search_text="Unknown")

    assert result.grounded is False
    assert provider.calls == 0


# ------------------------------------------------- healthcare boundary ----
def test_diagnosis_request_never_reaches_the_model(settings):
    """The boundary is enforced before generation, not by the prompt."""
    from app.rag.safety import CLINICAL_BOUNDARY_MESSAGE

    provider = FakeProvider()
    pipeline = build_pipeline(settings, [point(0, "haemoglobin", 0.9)], provider)

    result = pipeline.answer("My haemoglobin is 9. Do I have anaemia?")

    assert provider.calls == 0
    assert result.answer == CLINICAL_BOUNDARY_MESSAGE
    assert result.grounded is False
    assert result.finish_reason == "clinical_boundary"
    assert result.hits == []


def test_treatment_request_never_reaches_the_model(settings):
    provider = FakeProvider()
    pipeline = build_pipeline(settings, [point(0, "text", 0.9)], provider)

    pipeline.answer("What treatment should I take for high cholesterol?")

    assert provider.calls == 0


def test_informational_question_is_not_blocked(settings):
    provider = FakeProvider()
    pipeline = build_pipeline(settings, [point(0, "red cells", 0.9)], provider)

    result = pipeline.answer("What does a CBC test measure?")

    assert provider.calls == 1
    assert result.grounded is True


def test_clinical_advice_in_the_answer_is_suppressed(settings):
    """Defence in depth: the model crossed the line despite the system prompt."""
    from app.rag.safety import CLINICAL_BOUNDARY_MESSAGE

    provider = FakeProvider(
        answer="Yes, your haemoglobin level of 9 indicates mild anemia. "
        "Iron supplements may be recommended."
    )
    pipeline = build_pipeline(settings, [point(0, "haemoglobin", 0.9)], provider)

    result = pipeline.answer("What is haemoglobin?")

    assert provider.calls == 1
    assert result.answer == CLINICAL_BOUNDARY_MESSAGE
    assert result.grounded is False
    # Sources are dropped too -- they would lend authority to a suppressed answer.
    assert result.hits == []
