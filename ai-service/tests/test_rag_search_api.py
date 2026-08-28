"""POST /rag/search endpoint tests."""

import pytest

from app.rag.embeddings import get_embedder
from app.rag.vector_store import VectorStore, get_vector_store
from tests.test_ingestion import StubEmbedder
from tests.test_vector_store import StubClient, StubPoint, make_chunk


@pytest.fixture(name="search_client")
def search_client_fixture(client, settings):
    stub = StubClient(existing=True)
    store = VectorStore(settings=settings, client=stub)
    client.app.dependency_overrides[get_vector_store] = lambda: store
    client.app.dependency_overrides[get_embedder] = lambda: StubEmbedder()
    return client, stub


def hit(index: int, text: str, score: float, document_id: str = "cbc") -> StubPoint:
    payload = make_chunk(index, document_id=document_id).to_payload()
    payload["text"] = text
    return StubPoint(payload, score=score)


# ------------------------------------------------------------------ happy ---
def test_search_returns_ranked_results(search_client):
    client, stub = search_client
    stub.points = [hit(0, "red cells", 0.91), hit(1, "platelets", 0.72)]

    response = client.post(
        "/rag/search", json={"query": "What does a CBC test measure?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "What does a CBC test measure?"
    assert body["retrieval_count"] == 2
    assert [row["text"] for row in body["results"]] == ["red cells", "platelets"]


def test_results_carry_citation_fields(search_client):
    client, stub = search_client
    stub.points = [hit(0, "red cells", 0.91)]

    row = client.post("/rag/search", json={"query": "cbc"}).json()["results"][0]

    assert set(row) >= {"text", "score", "source", "title", "chunk_id"}
    assert row["source"] == "cbc.md"
    assert row["chunk_id"] == "cbc::0"
    assert row["score"] == pytest.approx(0.91)


def test_response_reports_applied_settings(search_client, settings):
    client, stub = search_client
    stub.points = []

    body = client.post("/rag/search", json={"query": "cbc"}).json()

    assert body["top_k"] == settings.top_k
    assert body["score_threshold"] == settings.score_threshold


def test_response_includes_timings(search_client):
    client, stub = search_client
    stub.points = [hit(0, "text", 0.5)]

    timings = client.post("/rag/search", json={"query": "cbc"}).json()["timings"]

    assert timings["embed_ms"] >= 0
    assert timings["search_ms"] >= 0
    assert timings["total_ms"] >= 0


def test_no_matches_returns_200_with_empty_results(search_client):
    client, stub = search_client
    stub.points = []

    response = client.post("/rag/search", json={"query": "quantum chromodynamics"})

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["retrieval_count"] == 0


# ---------------------------------------------------------------- options ---
def test_top_k_override_is_applied(search_client):
    client, stub = search_client
    stub.points = []

    body = client.post("/rag/search", json={"query": "cbc", "top_k": 3}).json()

    assert stub.queries[0]["limit"] == 3
    assert body["top_k"] == 3


def test_score_threshold_override_is_applied(search_client):
    client, stub = search_client
    stub.points = []

    client.post("/rag/search", json={"query": "cbc", "score_threshold": 0.8})

    assert stub.queries[0]["score_threshold"] == 0.8


def test_category_filter_is_applied(search_client):
    client, stub = search_client
    stub.points = []

    client.post("/rag/search", json={"query": "cbc", "category": "sop"})

    condition = stub.queries[0]["query_filter"].must[0]
    assert condition.key == "category"
    assert condition.match.value == "sop"


def test_source_filter_is_applied(search_client):
    client, stub = search_client
    stub.points = []

    client.post("/rag/search", json={"query": "x", "source": "patient_faq.md"})

    assert stub.queries[0]["query_filter"].must[0].key == "source"


def test_no_filter_is_sent_when_none_requested(search_client):
    client, stub = search_client
    stub.points = []

    client.post("/rag/search", json={"query": "cbc"})

    assert stub.queries[0]["query_filter"] is None


# ------------------------------------------------------------- validation ---
def test_blank_query_is_rejected(search_client):
    client, _ = search_client

    response = client.post("/rag/search", json={"query": "   "})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "EMPTY_QUERY"


def test_missing_query_is_a_validation_error(search_client):
    client, _ = search_client

    response = client.post("/rag/search", json={})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_empty_string_query_is_rejected_by_schema(search_client):
    client, _ = search_client

    assert client.post("/rag/search", json={"query": ""}).status_code == 422


def test_overlong_query_is_rejected(search_client):
    client, _ = search_client

    response = client.post("/rag/search", json={"query": "a" * 1001})

    assert response.status_code == 422


def test_top_k_bounds_are_enforced(search_client):
    client, _ = search_client

    assert client.post("/rag/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert (
        client.post("/rag/search", json={"query": "x", "top_k": 21}).status_code == 422
    )


def test_score_threshold_bounds_are_enforced(search_client):
    client, _ = search_client

    assert (
        client.post(
            "/rag/search", json={"query": "x", "score_threshold": 1.5}
        ).status_code
        == 422
    )


# ------------------------------------------------------------- resilience ---
def test_vector_store_outage_returns_503(client, settings):
    stub = StubClient(existing=True, fail_on="query_points")
    client.app.dependency_overrides[get_vector_store] = lambda: VectorStore(
        settings=settings, client=stub
    )
    client.app.dependency_overrides[get_embedder] = lambda: StubEmbedder()

    response = client.post("/rag/search", json={"query": "cbc"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "VECTOR_STORE_UNAVAILABLE"


def test_search_never_loads_the_generation_model(search_client):
    from app.core import runtime

    client, stub = search_client
    stub.points = [hit(0, "text", 0.5)]

    client.post("/rag/search", json={"query": "cbc"})

    assert runtime.is_loaded(runtime.GENERATION) is False


def test_request_id_is_echoed(search_client):
    client, stub = search_client
    stub.points = []

    response = client.post(
        "/rag/search", json={"query": "cbc"}, headers={"X-Request-ID": "trace-1"}
    )

    assert response.headers["X-Request-ID"] == "trace-1"
