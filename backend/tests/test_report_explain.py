"""POST /api/reports/{id}/explain tests.

The most important assertion inspects the actual HTTP request body sent to the
AI service and asserts no patient identifier is in it.
"""

import json

import httpx
import pytest
import respx

from app.services.ai_client import AIServiceClient, get_ai_client

AI_BASE = "http://ai-service:8001"

EXPLANATION = {
    "answer": "A complete blood count measures red cells, white cells and platelets [1].",
    "sources": [
        {
            "title": "Complete Blood Count (CBC) Guide",
            "source": "cbc.md",
            "chunk_id": "lab_tests/cbc::0",
            "score": 0.66,
            "section": "What a CBC measures",
            "category": "lab_tests",
        }
    ],
    "retrieval_count": 1,
    "grounded": True,
    "disclaimer": "AI-generated informational explanation. Not a medical diagnosis.",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "provider": "local",
    "finish_reason": "stop",
    "prompt_truncated": False,
    "timings": {"retrieval_ms": 10.0, "generation_ms": 800.0, "total_ms": 810.0},
}

IDENTIFYING_PAYLOAD = {
    "patient_name": "Asha Nair",
    "age": 34,
    "gender": "Female",
    "phone": "+91 98765 10001",
    "email": "asha.nair@example.com",
    "city": "Mumbai",
    "lab_branch": "BKC Flagship",
    "test_type": "CBC Panel",
    "doctor_name": "Dr. Naina Rao",
    "status": "processing",
    "priority": "urgent",
    "notes": "Call patient on 9876543210. Sister is Priya.",
}


@pytest.fixture(name="explain_client")
def explain_client_fixture(client):
    client.app.dependency_overrides[get_ai_client] = lambda: AIServiceClient()
    return client


@pytest.fixture(name="report_id")
def report_id_fixture(explain_client):
    created = explain_client.post("/api/reports", json=IDENTIFYING_PAYLOAD)
    assert created.status_code == 201
    return created.json()["id"]


# ------------------------------------------------------------ PII boundary --
@respx.mock
def test_no_identifier_reaches_the_ai_service(explain_client, report_id):
    route = respx.post(f"{AI_BASE}/rag/explain").mock(
        return_value=httpx.Response(200, json=EXPLANATION)
    )

    explain_client.post(f"/api/reports/{report_id}/explain")

    sent = route.calls[0].request.content.decode()
    for identifier in (
        "Asha Nair",
        "asha.nair@example.com",
        "98765 10001",
        "9876543210",
        "Naina Rao",
        "Priya",
        "Female",
    ):
        assert identifier not in sent, f"{identifier!r} leaked to the AI service"


@respx.mock
def test_exact_age_does_not_reach_the_ai_service(explain_client, report_id):
    route = respx.post(f"{AI_BASE}/rag/explain").mock(
        return_value=httpx.Response(200, json=EXPLANATION)
    )

    explain_client.post(f"/api/reports/{report_id}/explain")

    summary = json.loads(route.calls[0].request.content)["report_summary"]
    assert "34" not in summary
    assert "age_group: 30-39" in summary


@respx.mock
def test_only_operational_fields_are_sent(explain_client, report_id):
    route = respx.post(f"{AI_BASE}/rag/explain").mock(
        return_value=httpx.Response(200, json=EXPLANATION)
    )

    explain_client.post(f"/api/reports/{report_id}/explain")

    summary = json.loads(route.calls[0].request.content)["report_summary"]
    keys = {line.split(":")[0] for line in summary.splitlines()}
    assert keys <= {
        "test_type",
        "status",
        "priority",
        "city",
        "branch",
        "age_group",
        "due_status",
    }


@respx.mock
def test_search_text_is_the_test_name(explain_client, report_id):
    route = respx.post(f"{AI_BASE}/rag/explain").mock(
        return_value=httpx.Response(200, json=EXPLANATION)
    )

    explain_client.post(f"/api/reports/{report_id}/explain")

    assert json.loads(route.calls[0].request.content)["search_text"] == "CBC Panel"


@respx.mock
def test_context_sent_is_returned_for_audit(explain_client, report_id):
    respx.post(f"{AI_BASE}/rag/explain").mock(
        return_value=httpx.Response(200, json=EXPLANATION)
    )

    body = explain_client.post(f"/api/reports/{report_id}/explain").json()

    assert body["context_sent"]["test_type"] == "CBC Panel"
    assert body["context_sent"]["age_group"] == "30-39"
    assert "patient_name" not in body["context_sent"]


# ----------------------------------------------------------------- happy ---
@respx.mock
def test_explanation_is_returned_with_sources(explain_client, report_id):
    respx.post(f"{AI_BASE}/rag/explain").mock(
        return_value=httpx.Response(200, json=EXPLANATION)
    )

    response = explain_client.post(f"/api/reports/{report_id}/explain")

    assert response.status_code == 200
    body = response.json()
    assert body["report_id"] == report_id
    assert body["test_type"] == "CBC Panel"
    assert body["grounded"] is True
    assert body["sources"][0]["source"] == "cbc.md"
    assert "not a medical diagnosis" in body["disclaimer"].lower()


@respx.mock
def test_top_k_option_is_forwarded(explain_client, report_id):
    route = respx.post(f"{AI_BASE}/rag/explain").mock(
        return_value=httpx.Response(200, json=EXPLANATION)
    )

    explain_client.post(f"/api/reports/{report_id}/explain", json={"top_k": 3})

    assert json.loads(route.calls[0].request.content)["top_k"] == 3


@respx.mock
def test_request_id_is_propagated(explain_client, report_id):
    route = respx.post(f"{AI_BASE}/rag/explain").mock(
        return_value=httpx.Response(200, json=EXPLANATION)
    )

    explain_client.post(
        f"/api/reports/{report_id}/explain", headers={"X-Request-ID": "explain-trace"}
    )

    assert route.calls[0].request.headers["X-Request-ID"] == "explain-trace"


# -------------------------------------------------------------- failures ---
def test_unknown_report_returns_404(explain_client):
    response = explain_client.post("/api/reports/999999/explain")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "REPORT_NOT_FOUND"


@respx.mock
def test_unknown_report_never_calls_the_ai_service(explain_client):
    route = respx.post(f"{AI_BASE}/rag/explain").mock(
        return_value=httpx.Response(200, json=EXPLANATION)
    )

    explain_client.post("/api/reports/999999/explain")

    assert route.call_count == 0


@respx.mock
def test_ai_outage_returns_503_and_leaves_the_report_intact(explain_client, report_id):
    respx.post(f"{AI_BASE}/rag/explain").mock(
        side_effect=httpx.ConnectError("refused")
    )

    explained = explain_client.post(f"/api/reports/{report_id}/explain")
    fetched = explain_client.get(f"/api/reports/{report_id}")

    assert explained.status_code == 503
    assert explained.json()["error"]["code"] == "AI_SERVICE_UNAVAILABLE"
    assert fetched.status_code == 200
    assert fetched.json()["patient_name"] == "Asha Nair"


@respx.mock
def test_generation_disabled_is_forwarded(explain_client, report_id):
    respx.post(f"{AI_BASE}/rag/explain").mock(
        return_value=httpx.Response(
            503, json={"error": {"code": "GENERATION_DISABLED", "message": "off"}}
        )
    )

    response = explain_client.post(f"/api/reports/{report_id}/explain")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "GENERATION_DISABLED"


@respx.mock
def test_ungrounded_explanation_is_reported_honestly(explain_client, report_id):
    respx.post(f"{AI_BASE}/rag/explain").mock(
        return_value=httpx.Response(
            200,
            json={
                **EXPLANATION,
                "grounded": False,
                "sources": [],
                "retrieval_count": 0,
                "finish_reason": "no_context",
            },
        )
    )

    body = explain_client.post(f"/api/reports/{report_id}/explain").json()

    assert body["grounded"] is False
    assert body["sources"] == []
