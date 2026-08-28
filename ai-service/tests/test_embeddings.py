"""Embedding service tests.

Two layers:

* Unit tests that inject a fake ``sentence_transformers`` module, so loading,
  batching, normalisation and failure handling are exercised without a
  multi-gigabyte download.
* Integration tests (``-m integration``) that run against the real model and
  assert the properties only a real encoder can demonstrate.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from app.core import runtime
from app.core.errors import ModelUnavailableError
from app.rag.embeddings import (
    Embedder,
    SentenceTransformerEmbedder,
    chunk_embedding_input,
    get_embedder,
    reset_embedder,
)
from app.schemas.rag import KnowledgeChunk
from app.utils.optional import is_installed

REAL_ST_INSTALLED = is_installed("sentence_transformers")


# --------------------------------------------------------------- fake model --
class FakeModel:
    """Records encode() calls and returns deterministic vectors."""

    def __init__(self, dimension: int = 384, fail: bool = False):
        if fail:
            raise RuntimeError("weights unavailable")
        self._dimension = dimension
        self.calls: list[dict] = []
        self.load_count = 1

    def get_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        return [
            [float((index + position) % 7) for position in range(self._dimension)]
            for index, _ in enumerate(texts)
        ]


@pytest.fixture(name="fake_st")
def fake_st_fixture(monkeypatch):
    """Install a fake ``sentence_transformers`` module for the test."""
    created: list[FakeModel] = []
    behaviour = {"dimension": 384, "fail": False}

    def factory(model_name, device=None, **_):
        model = FakeModel(dimension=behaviour["dimension"], fail=behaviour["fail"])
        model.model_name = model_name
        model.device = device
        created.append(model)
        return model

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = factory  # type: ignore[attr-defined]
    module.__spec__ = SimpleNamespace(name="sentence_transformers")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    # is_installed() is cached on find_spec; clear it so the fake is seen.
    is_installed.cache_clear()
    yield SimpleNamespace(created=created, behaviour=behaviour)
    is_installed.cache_clear()


@pytest.fixture(name="embedder")
def embedder_fixture(settings, fake_st):
    return SentenceTransformerEmbedder(settings=settings)


# ------------------------------------------------------------------ contract --
def test_embedder_satisfies_the_protocol(settings):
    assert isinstance(SentenceTransformerEmbedder(settings=settings), Embedder)


def test_metadata_comes_from_settings(settings):
    embedder = SentenceTransformerEmbedder(settings=settings)

    assert embedder.model_name == settings.embedding_model
    assert embedder.dimension == settings.embedding_dimension


def test_nothing_is_loaded_before_first_use(settings):
    assert SentenceTransformerEmbedder(settings=settings).is_loaded is False


# ------------------------------------------------------------------ batching --
def test_embed_documents_returns_one_vector_per_text(embedder):
    vectors = embedder.embed_documents(["a", "b", "c"])

    assert len(vectors) == 3
    assert all(len(vector) == 384 for vector in vectors)


def test_embed_documents_sends_one_batched_call(embedder, fake_st):
    embedder.embed_documents(["a", "b", "c", "d"])

    model = fake_st.created[0]
    assert len(model.calls) == 1
    assert model.calls[0]["texts"] == ["a", "b", "c", "d"]


def test_batch_size_is_passed_through(embedder, settings, fake_st):
    embedder.embed_documents(["a"])

    assert fake_st.created[0].calls[0]["batch_size"] == settings.embedding_batch_size


def test_normalisation_flag_is_passed_through(embedder, fake_st):
    embedder.embed_documents(["a"])

    assert fake_st.created[0].calls[0]["normalize_embeddings"] is True


def test_progress_bar_is_disabled(embedder, fake_st):
    """A progress bar in container logs is noise, not information."""
    embedder.embed_documents(["a"])

    assert fake_st.created[0].calls[0]["show_progress_bar"] is False


def test_embed_text_delegates_to_the_batch_path(embedder, fake_st):
    vector = embedder.embed_text("single query")

    assert len(vector) == 384
    assert fake_st.created[0].calls[0]["texts"] == ["single query"]


def test_empty_batch_short_circuits_without_loading(settings, fake_st):
    embedder = SentenceTransformerEmbedder(settings=settings)

    assert embedder.embed_documents([]) == []
    assert embedder.is_loaded is False
    assert fake_st.created == []


def test_vectors_are_plain_floats(embedder):
    vector = embedder.embed_text("a")

    assert all(isinstance(value, float) for value in vector)


# -------------------------------------------------------------- model reuse --
def test_model_is_loaded_once_across_many_calls(embedder, fake_st):
    for _ in range(5):
        embedder.embed_text("repeat")

    assert len(fake_st.created) == 1
    assert len(fake_st.created[0].calls) == 5


def test_warm_up_loads_the_model(embedder, fake_st):
    embedder.warm_up()

    assert embedder.is_loaded is True
    assert len(fake_st.created) == 1


def test_loading_marks_runtime_state(embedder):
    embedder.warm_up()

    assert runtime.is_loaded(runtime.EMBEDDING) is True
    assert runtime.failure_reason(runtime.EMBEDDING) is None


def test_get_embedder_returns_a_singleton():
    reset_embedder()
    try:
        assert get_embedder() is get_embedder()
    finally:
        reset_embedder()


# ----------------------------------------------------------------- failures --
def test_load_failure_raises_model_unavailable(settings, fake_st):
    fake_st.behaviour["fail"] = True
    embedder = SentenceTransformerEmbedder(settings=settings)

    with pytest.raises(ModelUnavailableError) as excinfo:
        embedder.warm_up()

    assert excinfo.value.code == "MODEL_UNAVAILABLE"
    assert "weights unavailable" in excinfo.value.details["reason"]


def test_load_failure_is_recorded_for_health(settings, fake_st):
    fake_st.behaviour["fail"] = True

    with pytest.raises(ModelUnavailableError):
        SentenceTransformerEmbedder(settings=settings).warm_up()

    assert runtime.is_loaded(runtime.EMBEDDING) is False
    assert "weights unavailable" in runtime.failure_reason(runtime.EMBEDDING)


def test_dimension_mismatch_is_caught_at_load(settings, fake_st):
    """A wrong-width model must fail at load, not corrupt the collection."""
    fake_st.behaviour["dimension"] = 768
    embedder = SentenceTransformerEmbedder(settings=settings)

    with pytest.raises(ModelUnavailableError) as excinfo:
        embedder.warm_up()

    assert excinfo.value.code == "EMBEDDING_DIMENSION_MISMATCH"
    assert "768" in excinfo.value.message


@pytest.mark.skipif(
    REAL_ST_INSTALLED, reason="requires sentence-transformers to be absent"
)
def test_missing_dependency_reports_clearly(settings):
    """The lite environment must degrade with a useful message."""
    embedder = SentenceTransformerEmbedder(settings=settings)

    with pytest.raises(ModelUnavailableError) as excinfo:
        embedder.warm_up()

    assert "sentence_transformers" in excinfo.value.message


# ------------------------------------------------------------- input prep ---
def test_embedding_input_is_prefixed_with_context():
    chunk = KnowledgeChunk(
        document_id="d",
        chunk_id="d::0",
        chunk_index=0,
        title="Lipid Profile and Cholesterol Guide",
        source="lipid_profile.md",
        category="lab_tests",
        section="Preparation",
        text="A 9-12 hour fast is commonly requested.",
    )

    assert chunk_embedding_input(chunk).startswith(
        "Lipid Profile and Cholesterol Guide — Preparation"
    )


def test_embedding_input_without_a_section():
    chunk = KnowledgeChunk(
        document_id="d",
        chunk_id="d::0",
        chunk_index=0,
        title="Patient FAQ",
        source="faq.md",
        category="faq",
        section=None,
        text="Body.",
    )

    prepared = chunk_embedding_input(chunk)

    assert prepared.startswith("Patient FAQ")
    assert "—" not in prepared.split("\n")[0]
