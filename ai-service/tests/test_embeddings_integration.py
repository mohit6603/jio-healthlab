"""Embedding tests against the real Sentence-Transformers model.

Skipped unless sentence-transformers is installed, so the lite unit run stays
fast. Inside the ai-service image::

    pytest -m integration tests/test_embeddings_integration.py
"""

from __future__ import annotations

import math

import pytest

from app.config import Settings
from app.core import runtime
from app.rag.embeddings import SentenceTransformerEmbedder
from app.utils.optional import is_installed

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not is_installed("sentence_transformers"),
        reason="sentence-transformers is not installed",
    ),
]


@pytest.fixture(name="real_embedder", scope="module")
def real_embedder_fixture():
    """One model load for the whole module -- loading is the expensive part."""
    embedder = SentenceTransformerEmbedder(settings=Settings(log_json=False))
    embedder.warm_up()
    return embedder


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


# ------------------------------------------------------------------ shape ---
def test_real_model_emits_configured_dimension(real_embedder):
    assert len(real_embedder.embed_text("complete blood count")) == 384


def test_vectors_are_unit_length(real_embedder):
    """Cosine distance in Qdrant assumes normalised vectors."""
    for text in ["platelets", "a much longer sentence about thyroid hormones"]:
        assert norm(real_embedder.embed_text(text)) == pytest.approx(1.0, abs=1e-5)


def test_this_model_normalises_internally(real_embedder):
    """all-MiniLM-L6-v2 ends its pipeline with a Normalize module.

    Output is therefore unit-length whether or not ``normalize_embeddings`` is
    set. The explicit flag is kept anyway: it makes the unit-length guarantee a
    property of *our* embedder rather than of whichever model is configured, so
    swapping to a model without a Normalize module cannot silently break cosine
    similarity in Qdrant.
    """
    unflagged = SentenceTransformerEmbedder(
        settings=Settings(embedding_normalize=False, log_json=False)
    )

    assert norm(unflagged.embed_text("platelets")) == pytest.approx(1.0, abs=1e-5)


def test_embed_text_matches_single_element_batch(real_embedder):
    single = real_embedder.embed_text("kidney function")
    batched = real_embedder.embed_documents(["kidney function"])[0]

    assert single == pytest.approx(batched, abs=1e-6)


def test_batch_order_is_preserved(real_embedder):
    texts = ["platelets", "thyroid hormone", "creatinine"]

    batch = real_embedder.embed_documents(texts)

    for text, vector in zip(texts, batch, strict=True):
        assert cosine(vector, real_embedder.embed_text(text)) == pytest.approx(
            1.0, abs=1e-5
        )


def test_embedding_is_deterministic(real_embedder):
    first = real_embedder.embed_text("liver enzymes")
    second = real_embedder.embed_text("liver enzymes")

    assert first == pytest.approx(second, abs=1e-6)


# ------------------------------------------------------------- semantics ---
def test_related_text_scores_higher_than_unrelated(real_embedder):
    query = real_embedder.embed_text("What does a CBC test measure?")
    related = real_embedder.embed_text(
        "A complete blood count measures red cells, white cells and platelets."
    )
    unrelated = real_embedder.embed_text(
        "Courier runs depart the branch on a fixed schedule."
    )

    related_score = cosine(query, related)
    unrelated_score = cosine(query, unrelated)

    # Ranking is what retrieval depends on; the absolute value is model- and
    # length-dependent, so assert a clear margin rather than a magic threshold.
    assert related_score > unrelated_score + 0.2
    assert related_score > 0.3


def test_paraphrases_cluster_together(real_embedder):
    a = real_embedder.embed_text("Do I need to fast before this blood test?")
    b = real_embedder.embed_text("Is fasting required before the sample is taken?")
    c = real_embedder.embed_text("Which cities have collection branches?")

    assert cosine(a, b) > cosine(a, c)


# ------------------------------------------------------------ truncation ---
def test_input_beyond_the_window_is_truncated(real_embedder):
    """Evidence for the CHUNK_SIZE default.

    Text past the model's 256 word-piece window does not change the vector at
    all -- which is exactly why chunks are budgeted below that limit rather
    than at the 500 tokens a naive configuration would use.
    """
    prefix = "The complete blood count measures circulating cells. " * 40
    extended = prefix + " ".join(f"Distinct trailing sentence {i}." for i in range(200))

    assert cosine(
        real_embedder.embed_text(prefix), real_embedder.embed_text(extended)
    ) == pytest.approx(1.0, abs=1e-4)


def test_short_text_within_the_window_is_not_truncated(real_embedder):
    short = "The complete blood count measures circulating cells."
    extended = short + " It also reports haemoglobin and haematocrit."

    assert cosine(
        real_embedder.embed_text(short), real_embedder.embed_text(extended)
    ) < 0.99


# ----------------------------------------------------------- model reuse ---
def test_runtime_reports_the_model_as_loaded(real_embedder):
    runtime.mark_loaded(runtime.EMBEDDING)

    assert runtime.is_loaded(runtime.EMBEDDING) is True


def test_second_use_does_not_reload(real_embedder):
    import time

    real_embedder.embed_text("warm")
    started = time.perf_counter()
    real_embedder.embed_text("already loaded")
    elapsed = time.perf_counter() - started

    # A reload would take seconds; a cached model encodes in milliseconds.
    assert elapsed < 1.0
