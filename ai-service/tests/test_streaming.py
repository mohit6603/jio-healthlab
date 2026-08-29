"""Streaming RAG tests.

The refusal paths matter most here: a streamed answer must refuse for exactly
the same reasons and at exactly the same points as the non-streaming one, or
the safety guarantees differ depending on which endpoint a client happens to
call.
"""

from __future__ import annotations

import json

import pytest

from app.llm import get_llm_provider
from app.llm.provider import DisabledProvider, GenerationResult, LLMProvider
from app.rag.embeddings import get_embedder
from app.rag.pipeline import RagPipeline
from app.rag.prompts import INSUFFICIENT_CONTEXT_MESSAGE
from app.rag.retriever import Retriever
from app.rag.safety import CLINICAL_BOUNDARY_MESSAGE
from app.rag.vector_store import VectorStore, get_vector_store
from tests.test_ingestion import StubEmbedder
from tests.test_vector_store import StubClient, StubPoint, make_chunk


class StreamingProvider(LLMProvider):
    """Yields a scripted answer one word at a time."""

    name = "fake-stream"

    def __init__(self, answer: str = "A CBC measures blood cells."):
        self._answer = answer
        self.stream_calls = 0
        self.generate_calls = 0

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def is_available(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    def generate(self, prompt, *, system=None, max_new_tokens=None, temperature=None):
        self.generate_calls += 1
        return GenerationResult(text=self._answer, provider=self.name, model="fake-model")

    def stream(self, prompt, *, system=None, max_new_tokens=None, temperature=None):
        self.stream_calls += 1
        for word in self._answer.split(" "):
            yield f"{word} "


def point(index: int, text: str, score: float = 0.9) -> StubPoint:
    payload = make_chunk(index).to_payload()
    payload["text"] = text
    return StubPoint(payload, score=score)


def build(settings, points, provider=None):
    client = StubClient(existing=True)
    client.points = points
    store = VectorStore(settings=settings, client=client)
    return RagPipeline(
        Retriever(StubEmbedder(), store, settings),
        provider or StreamingProvider(),
        settings,
    )


def collect(pipeline, question: str):
    return list(pipeline.stream_answer(question))


# ------------------------------------------------------------ event order --
def test_sources_arrive_before_the_first_token(settings):
    events = collect(build(settings, [point(0, "red cells")]), "What is a CBC?")

    kinds = [event.type for event in events]
    assert kinds[0] == "sources"
    assert kinds[1] == "token"
    assert kinds[-1] == "done"


def test_exactly_one_done_event(settings):
    events = collect(build(settings, [point(0, "red cells")]), "What is a CBC?")

    assert sum(1 for e in events if e.type == "done") == 1


def test_tokens_reassemble_into_the_answer(settings):
    events = collect(build(settings, [point(0, "red cells")]), "What is a CBC?")

    text = "".join(e.text for e in events if e.type == "token").strip()
    assert text == "A CBC measures blood cells."


def test_sources_event_carries_the_hits(settings):
    events = collect(build(settings, [point(0, "red cells")]), "What is a CBC?")

    sources = next(e for e in events if e.type == "sources")
    assert len(sources.hits) == 1
    assert sources.hits[0].source == "cbc.md"


def test_done_reports_grounding_and_model(settings):
    events = collect(build(settings, [point(0, "red cells")]), "What is a CBC?")

    done = events[-1]
    assert done.grounded is True
    assert done.model == "fake-model"
    assert done.retrieval_count == 1
    assert done.total_ms >= 0


# --------------------------------------------------------------- refusals --
def test_no_context_streams_the_fixed_message_without_calling_the_model(settings):
    provider = StreamingProvider()
    events = collect(build(settings, [], provider), "who won the world cup")

    assert provider.stream_calls == 0
    assert provider.generate_calls == 0
    assert events[0].text == INSUFFICIENT_CONTEXT_MESSAGE
    assert events[-1].finish_reason == "no_context"
    assert events[-1].grounded is False


def test_clinical_question_never_reaches_the_model(settings):
    provider = StreamingProvider()
    events = collect(
        build(settings, [point(0, "haemoglobin")], provider),
        "My haemoglobin is 9. Do I have anaemia?",
    )

    assert provider.stream_calls == 0
    assert events[0].text == CLINICAL_BOUNDARY_MESSAGE
    assert events[-1].finish_reason == "clinical_boundary"


def test_no_sources_event_on_a_refusal(settings):
    events = collect(build(settings, []), "who won the world cup")

    assert not any(e.type == "sources" for e in events)


# ---------------------------------------------------- post-hoc suppression --
def test_clinical_answer_is_replaced_after_streaming(settings):
    """The screen can only run on the finished text, so it emits `replace`."""
    provider = StreamingProvider("You have anaemia based on these results.")
    events = collect(build(settings, [point(0, "haemoglobin")], provider), "What is haemoglobin?")

    replace = next(e for e in events if e.type == "replace")
    assert replace.text == CLINICAL_BOUNDARY_MESSAGE
    assert events[-1].drop_sources is True
    assert events[-1].grounded is False


def test_safe_answer_is_not_replaced(settings):
    events = collect(build(settings, [point(0, "red cells")]), "What is a CBC?")

    assert not any(e.type == "replace" for e in events)
    assert events[-1].drop_sources is False


def test_streamed_refusal_is_flagged_ungrounded(settings):
    provider = StreamingProvider("The reference material does not contain that.")
    events = collect(build(settings, [point(0, "text")], provider), "What is a CBC?")

    assert events[-1].grounded is False


# ------------------------------------------------------------- disabled ----
def test_disabled_provider_raises(settings):
    pipeline = build(settings, [point(0, "text")], DisabledProvider())

    with pytest.raises(Exception) as excinfo:
        collect(pipeline, "What is a CBC?")

    assert "GENERATION_DISABLED" in str(excinfo.value) or "disabled" in str(
        excinfo.value
    ).lower()


def test_default_stream_falls_back_to_generate(settings):
    """A provider without native streaming still satisfies the interface."""

    class NonStreaming(LLMProvider):
        name = "plain"

        @property
        def model_name(self) -> str:
            return "plain-model"

        @property
        def is_available(self) -> bool:
            return True

        def generate(self, prompt, *, system=None, max_new_tokens=None, temperature=None):
            return GenerationResult(
                text="whole answer", provider="plain", model="plain-model"
            )

    provider = NonStreaming()
    assert provider.supports_streaming is False
    assert list(provider.stream("q")) == ["whole answer"]


# ------------------------------------------------------------------ SSE ----
@pytest.fixture(name="stream_client")
def stream_client_fixture(client, settings):
    stub = StubClient(existing=True)
    stub.points = [point(0, "red cells")]
    store = VectorStore(settings=settings, client=stub)
    provider = StreamingProvider()
    client.app.dependency_overrides[get_vector_store] = lambda: store
    client.app.dependency_overrides[get_embedder] = lambda: StubEmbedder()
    client.app.dependency_overrides[get_llm_provider] = lambda: provider
    return client


def parse_sse(body: str) -> list[tuple[str, dict]]:
    frames = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event, data = "message", []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if data:
            frames.append((event, json.loads("\n".join(data))))
    return frames


def test_sse_content_type(stream_client):
    response = stream_client.post("/rag/query/stream", json={"question": "cbc"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_sse_disables_proxy_buffering(stream_client):
    """Without this header nginx buffers the whole body and streaming is lost."""
    response = stream_client.post("/rag/query/stream", json={"question": "cbc"})

    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["cache-control"] == "no-cache"


def test_sse_frames_are_well_formed(stream_client):
    response = stream_client.post("/rag/query/stream", json={"question": "cbc"})
    frames = parse_sse(response.text)

    names = [name for name, _ in frames]
    assert names[0] == "sources"
    assert names[-1] == "done"


def test_sse_sources_payload(stream_client):
    frames = parse_sse(
        stream_client.post("/rag/query/stream", json={"question": "cbc"}).text
    )
    _, payload = frames[0]

    assert payload["sources"][0]["source"] == "cbc.md"
    assert "score" in payload["sources"][0]


def test_sse_done_payload_carries_the_disclaimer(stream_client):
    frames = parse_sse(
        stream_client.post("/rag/query/stream", json={"question": "cbc"}).text
    )
    _, done = frames[-1]

    assert "not a medical diagnosis" in done["disclaimer"].lower()
    assert done["grounded"] is True
    assert done["model"] == "fake-model"


def test_sse_blank_question_is_rejected(stream_client):
    response = stream_client.post("/rag/query/stream", json={"question": "  "})

    assert response.status_code == 422
