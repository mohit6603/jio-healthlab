"""Retriever unit tests against a stub embedder and stub vector store."""

import pytest

from app.core.errors import AIError, VectorStoreError
from app.rag.retriever import (
    MAX_QUERY_CHARS,
    EmptyQueryError,
    Retriever,
    build_filters,
    normalise_query,
)
from app.rag.vector_store import VectorStore
from app.schemas.rag import SearchHit
from tests.test_pipeline import StubEmbedder
from tests.test_vector_store import StubClient, StubPoint, make_chunk


@pytest.fixture(name="retriever_parts")
def retriever_parts_fixture(settings):
    client = StubClient(existing=True)
    store = VectorStore(settings=settings, client=client)
    embedder = StubEmbedder()
    return Retriever(embedder, store, settings), client, embedder


def payload(index: int, text: str, document_id: str = "cbc") -> dict:
    chunk = make_chunk(index, document_id=document_id)
    data = chunk.to_payload()
    data["text"] = text
    return data


# ------------------------------------------------------------ query rules --
def test_query_whitespace_is_collapsed():
    assert normalise_query("  What   does a  CBC measure? ") == (
        "What does a CBC measure?"
    )


def test_blank_query_is_rejected():
    with pytest.raises(EmptyQueryError) as excinfo:
        normalise_query("   \n\t  ")

    assert excinfo.value.code == "EMPTY_QUERY"


def test_overlong_query_is_rejected():
    with pytest.raises(AIError) as excinfo:
        normalise_query("word " * (MAX_QUERY_CHARS // 2))

    assert excinfo.value.code == "QUERY_TOO_LONG"


def test_query_at_the_limit_is_accepted():
    assert len(normalise_query("a" * MAX_QUERY_CHARS)) == MAX_QUERY_CHARS


# ---------------------------------------------------------------- filters --
def test_build_filters_drops_unset_fields():
    assert build_filters(category="lab_tests") == {"category": "lab_tests"}


def test_build_filters_returns_none_when_empty():
    assert build_filters() is None


def test_build_filters_combines_fields():
    assert build_filters(category="faq", source="patient_faq.md") == {
        "category": "faq",
        "source": "patient_faq.md",
    }


# -------------------------------------------------------------- retrieval --
def test_retrieve_returns_hits(retriever_parts):
    retriever, client, _ = retriever_parts
    client.points = [StubPoint(payload(0, "red cells"), score=0.9)]

    result = retriever.retrieve("What does a CBC measure?")

    assert result.count == 1
    assert result.hits[0].text == "red cells"
    assert result.query == "What does a CBC measure?"


def test_retrieve_embeds_the_cleaned_query(retriever_parts):
    retriever, _, embedder = retriever_parts

    retriever.retrieve("  spaced    out  ")

    assert embedder.calls[0] == ["spaced out"]


def test_retrieve_uses_configured_defaults(retriever_parts, settings):
    retriever, client, _ = retriever_parts

    result = retriever.retrieve("query")

    assert client.queries[0]["limit"] == settings.top_k
    assert client.queries[0]["score_threshold"] == settings.score_threshold
    assert result.top_k == settings.top_k


def test_retrieve_honours_explicit_overrides(retriever_parts):
    retriever, client, _ = retriever_parts

    result = retriever.retrieve("query", top_k=2, score_threshold=0.7)

    assert client.queries[0]["limit"] == 2
    assert client.queries[0]["score_threshold"] == 0.7
    assert result.score_threshold == 0.7


def test_zero_threshold_is_not_treated_as_unset(retriever_parts):
    """0.0 is a meaningful threshold and must not fall back to the default."""
    retriever, client, _ = retriever_parts

    retriever.retrieve("query", score_threshold=0.0)

    assert client.queries[0]["score_threshold"] == 0.0


def test_retrieve_applies_filters(retriever_parts):
    retriever, client, _ = retriever_parts

    retriever.retrieve("query", filters={"category": "sop"})

    condition = client.queries[0]["query_filter"].must[0]
    assert condition.key == "category"


def test_results_are_ranked_best_first(retriever_parts):
    retriever, client, _ = retriever_parts
    client.points = [
        StubPoint(payload(0, "low"), score=0.30),
        StubPoint(payload(1, "high"), score=0.95),
        StubPoint(payload(2, "mid"), score=0.60),
    ]

    result = retriever.retrieve("query")

    assert [hit.text for hit in result.hits] == ["high", "mid", "low"]
    assert result.best_score == 0.95


def test_no_matches_is_an_empty_result_not_an_error(retriever_parts):
    retriever, client, _ = retriever_parts
    client.points = []

    result = retriever.retrieve("something unrelated")

    assert result.is_empty is True
    assert result.count == 0
    assert result.best_score is None


def test_timings_are_recorded(retriever_parts):
    retriever, client, _ = retriever_parts
    client.points = [StubPoint(payload(0, "text"))]

    result = retriever.retrieve("query")

    assert result.embed_ms >= 0
    assert result.search_ms >= 0
    assert result.total_ms >= result.search_ms


def test_sources_are_deduplicated_in_rank_order(retriever_parts):
    retriever, client, _ = retriever_parts
    client.points = [
        StubPoint(payload(0, "a", "cbc"), score=0.9),
        StubPoint(payload(1, "b", "cbc"), score=0.8),
        StubPoint(payload(0, "c", "thyroid"), score=0.7),
    ]

    result = retriever.retrieve("query")

    assert result.sources() == ["cbc.md", "thyroid.md"]


def test_vector_store_failure_propagates(settings):
    client = StubClient(existing=True, fail_on="query_points")
    retriever = Retriever(
        StubEmbedder(), VectorStore(settings=settings, client=client), settings
    )

    with pytest.raises(VectorStoreError):
        retriever.retrieve("query")


def test_retrieval_does_not_touch_generation(retriever_parts):
    """Retrieval must never require an LLM -- Rule 16."""
    from app.core import runtime

    retriever, client, _ = retriever_parts
    client.points = [StubPoint(payload(0, "text"))]

    retriever.retrieve("query")

    assert runtime.is_loaded(runtime.GENERATION) is False


def test_hits_carry_full_citation_metadata(retriever_parts):
    retriever, client, _ = retriever_parts
    client.points = [StubPoint(payload(0, "red cells"), score=0.9)]

    hit = retriever.retrieve("query").hits[0]

    assert isinstance(hit, SearchHit)
    assert hit.source == "cbc.md"
    assert hit.title == "Complete Blood Count Guide"
    assert hit.chunk_id == "cbc::0"
    assert hit.document_id == "cbc"
    assert hit.category == "lab_tests"
    assert hit.section == "What it measures"
