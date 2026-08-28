"""Semantic report search tests.

The central guarantee: only sanitised text is embedded. Full records come from
the database, joined to the ids the vector store returns.
"""

import json

import httpx
import pytest
import respx

from app.models import Report
from app.services.ai_client import AIServiceClient, get_ai_client
from app.services.report_search_service import index_payload
from app.services.sanitizer import search_metadata, searchable_text

AI_BASE = "http://ai-service:8001"


def make_report(**overrides) -> Report:
    defaults = {
        "id": 7,
        "patient_name": "Asha Nair",
        "age": 34,
        "gender": "Female",
        "phone": "+91 98765 10001",
        "email": "asha.nair@example.com",
        "city": "Mumbai",
        "lab_branch": "Andheri Hub",
        "test_type": "Kidney Function",
        "doctor_name": "Dr. Naina Rao",
        "status": "processing",
        "priority": "urgent",
        "notes": "Call 9876543210. Sister is Priya.",
    }
    return Report(**{**defaults, **overrides})


# ------------------------------------------------------- searchable text ---
IDENTIFIERS = [
    "Asha Nair",
    "asha.nair@example.com",
    "98765 10001",
    "9876543210",
    "Naina Rao",
    "Priya",
    "Female",
]


@pytest.mark.parametrize("identifier", IDENTIFIERS)
def test_searchable_text_contains_no_identifier(identifier):
    assert identifier not in searchable_text(make_report())


def test_searchable_text_reads_as_a_sentence():
    text = searchable_text(make_report())

    assert text.startswith("Urgent Kidney Function test at the Andheri Hub branch")
    assert "in Mumbai" in text
    assert "Status: processing." in text


def test_searchable_text_uses_an_age_band_not_an_age():
    text = searchable_text(make_report(age=34))

    assert "34" not in text
    assert "30-39" in text


def test_urgent_reports_say_so():
    assert "urgent" in searchable_text(make_report(priority="urgent")).lower()


def test_routine_reports_are_not_marked_urgent():
    text = searchable_text(make_report(priority="routine"))

    assert "Marked urgent" not in text


def test_missing_branch_and_city_still_produce_text():
    text = searchable_text(make_report(lab_branch=None, city=None))

    assert text.startswith("Urgent Kidney Function test.")


def test_search_metadata_is_filterable_fields_only():
    metadata = search_metadata(make_report())

    assert set(metadata) == {"status", "priority", "branch", "city", "test_type"}
    assert metadata["branch"] == "Andheri Hub"


def test_search_metadata_omits_identifiers():
    assert "patient_name" not in search_metadata(make_report())


def test_index_payload_shape():
    payload = index_payload(make_report())

    assert payload["report_id"] == 7
    assert "Asha Nair" not in json.dumps(payload)


# ---------------------------------------------------------------- fixtures --
@pytest.fixture(name="search_client")
def search_client_fixture(client):
    client.app.dependency_overrides[get_ai_client] = lambda: AIServiceClient()
    return client


@pytest.fixture(name="seeded")
def seeded_fixture(search_client):
    rows = [
        {
            "patient_name": "Asha Nair",
            "age": 34,
            "test_type": "Kidney Function",
            "city": "Mumbai",
            "lab_branch": "Andheri Hub",
            "status": "processing",
            "priority": "urgent",
        },
        {
            "patient_name": "Rohan Kulkarni",
            "age": 52,
            "test_type": "Thyroid",
            "city": "Pune",
            "lab_branch": "Koregaon Park",
            "status": "registered",
            "priority": "routine",
        },
    ]
    return [search_client.post("/api/reports", json=row).json() for row in rows]


def hit(report_id: int, score: float, text: str = "summary") -> dict:
    return {"report_id": report_id, "score": score, "text": text}


# ------------------------------------------------------------- indexing ----
@respx.mock
def test_index_sends_only_sanitised_text(search_client, seeded):
    route = respx.post(f"{AI_BASE}/reports/index").mock(
        return_value=httpx.Response(
            201, json={"indexed": 2, "collection": "healthlab_reports"}
        )
    )

    search_client.post("/api/ai/report-index")

    sent = route.calls[0].request.content.decode()
    for identifier in ("Asha Nair", "Rohan Kulkarni"):
        assert identifier not in sent


@respx.mock
def test_index_reports_the_count_and_collection(search_client, seeded):
    respx.post(f"{AI_BASE}/reports/index").mock(
        return_value=httpx.Response(
            201, json={"indexed": 2, "collection": "healthlab_reports"}
        )
    )

    body = search_client.post("/api/ai/report-index").json()

    assert body["indexed"] == 2
    assert body["collection"] == "healthlab_reports"


@respx.mock
def test_index_sends_filter_metadata(search_client, seeded):
    route = respx.post(f"{AI_BASE}/reports/index").mock(
        return_value=httpx.Response(201, json={"indexed": 2, "collection": "c"})
    )

    search_client.post("/api/ai/report-index")

    items = json.loads(route.calls[0].request.content)["items"]
    assert set(items[0]["metadata"]) <= {
        "status",
        "priority",
        "branch",
        "city",
        "test_type",
    }


@respx.mock
def test_index_with_no_reports_calls_nothing(search_client):
    route = respx.post(f"{AI_BASE}/reports/index").mock(
        return_value=httpx.Response(201, json={"indexed": 0, "collection": "c"})
    )

    body = search_client.post("/api/ai/report-index").json()

    assert body["indexed"] == 0
    assert route.call_count == 0


# --------------------------------------------------------------- search ----
@respx.mock
def test_search_joins_hits_back_to_full_records(search_client, seeded):
    target = seeded[0]
    respx.post(f"{AI_BASE}/reports/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "query": "urgent kidney",
                "results": [hit(target["id"], 0.71, "Urgent Kidney Function test.")],
                "retrieval_count": 1,
            },
        )
    )

    body = search_client.post(
        "/api/ai/report-search", json={"query": "urgent kidney tests in Mumbai"}
    ).json()

    assert body["retrieval_count"] == 1
    match = body["results"][0]
    assert match["report"]["id"] == target["id"]
    # The full record, including the name, comes from the database -- not the
    # vector store.
    assert match["report"]["patient_name"] == "Asha Nair"
    assert match["score"] == pytest.approx(0.71)
    assert match["matched_summary"] == "Urgent Kidney Function test."


@respx.mock
def test_search_preserves_rank_order(search_client, seeded):
    first, second = seeded
    respx.post(f"{AI_BASE}/reports/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "query": "q",
                "results": [hit(second["id"], 0.9), hit(first["id"], 0.4)],
                "retrieval_count": 2,
            },
        )
    )

    body = search_client.post("/api/ai/report-search", json={"query": "q"}).json()

    assert [row["report"]["id"] for row in body["results"]] == [
        second["id"],
        first["id"],
    ]


@respx.mock
def test_filters_are_forwarded(search_client, seeded):
    route = respx.post(f"{AI_BASE}/reports/search").mock(
        return_value=httpx.Response(
            200, json={"query": "q", "results": [], "retrieval_count": 0}
        )
    )

    search_client.post(
        "/api/ai/report-search",
        json={"query": "kidney", "priority": "urgent", "city": "Mumbai", "top_k": 5},
    )

    sent = json.loads(route.calls[0].request.content)
    assert sent["priority"] == "urgent"
    assert sent["city"] == "Mumbai"
    assert sent["top_k"] == 5


@respx.mock
def test_unset_filters_are_omitted(search_client, seeded):
    route = respx.post(f"{AI_BASE}/reports/search").mock(
        return_value=httpx.Response(
            200, json={"query": "q", "results": [], "retrieval_count": 0}
        )
    )

    search_client.post("/api/ai/report-search", json={"query": "kidney"})

    assert json.loads(route.calls[0].request.content) == {"query": "kidney"}


@respx.mock
def test_stale_hits_are_dropped(search_client, seeded):
    """The index can lag a deletion; a hollow row is worse than no row."""
    respx.post(f"{AI_BASE}/reports/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "query": "q",
                "results": [hit(seeded[0]["id"], 0.9), hit(999999, 0.8)],
                "retrieval_count": 2,
            },
        )
    )

    body = search_client.post("/api/ai/report-search", json={"query": "q"}).json()

    assert body["retrieval_count"] == 1
    assert body["results"][0]["report"]["id"] == seeded[0]["id"]


@respx.mock
def test_no_matches_returns_an_empty_list(search_client, seeded):
    respx.post(f"{AI_BASE}/reports/search").mock(
        return_value=httpx.Response(
            200, json={"query": "q", "results": [], "retrieval_count": 0}
        )
    )

    body = search_client.post("/api/ai/report-search", json={"query": "q"}).json()

    assert body["results"] == []
    assert body["retrieval_count"] == 0


# ------------------------------------------------------------ resilience ---
@respx.mock
def test_index_unavailable_returns_503(search_client, seeded):
    respx.post(f"{AI_BASE}/reports/search").mock(
        return_value=httpx.Response(
            503,
            json={"error": {"code": "VECTOR_STORE_UNAVAILABLE", "message": "down"}},
        )
    )

    response = search_client.post("/api/ai/report-search", json={"query": "q"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "VECTOR_STORE_UNAVAILABLE"


@respx.mock
def test_keyword_search_still_works_when_semantic_search_is_down(
    search_client, seeded
):
    respx.post(f"{AI_BASE}/reports/search").mock(
        side_effect=httpx.ConnectError("refused")
    )

    semantic = search_client.post("/api/ai/report-search", json={"query": "kidney"})
    keyword = search_client.get("/api/reports", params={"search": "Kidney"})

    assert semantic.status_code == 503
    assert keyword.status_code == 200
    assert len(keyword.json()) == 1


def test_blank_query_is_rejected(search_client):
    assert (
        search_client.post("/api/ai/report-search", json={"query": ""}).status_code
        == 422
    )


def test_top_k_bounds_are_enforced(search_client):
    assert (
        search_client.post(
            "/api/ai/report-search", json={"query": "q", "top_k": 51}
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    ("test_type", "expected"),
    [
        ("Blood Test", "Routine Blood Test at"),
        ("CBC Panel", "Routine CBC Panel at"),
        ("MRI Scan", "Routine MRI Scan at"),
        ("Lipid Profile", "Routine Lipid Profile at"),
        ("Kidney Function", "Routine Kidney Function test at"),
        ("Thyroid", "Routine Thyroid test at"),
    ],
)
def test_test_name_reads_naturally(test_type, expected):
    """The summary is embedded, so it should read like language, not a dump."""
    text = searchable_text(make_report(test_type=test_type, priority="routine"))

    assert text.startswith(expected)
