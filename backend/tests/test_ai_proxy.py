"""AI proxy tests.

The AI service is mocked with respx: no model, no Qdrant, no network. What is
under test is the backend's translation of every AI-service outcome into
something the browser can act on -- and that an AI outage never touches the
reports system.
"""

import httpx
import pytest
import respx

from app.services.ai_client import AIServiceClient, get_ai_client

AI_BASE = "http://ai-service:8001"

ANSWER_PAYLOAD = {
    "answer": "A CBC measures red cells, white cells and platelets [1].",
    "sources": [
        {
            "title": "Complete Blood Count (CBC) Guide",
            "source": "cbc.md",
            "chunk_id": "lab_tests/cbc::0",
            "score": 0.664,
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
    "timings": {"retrieval_ms": 12.0, "generation_ms": 900.0, "total_ms": 912.0},
}

SEARCH_PAYLOAD = {
    "query": "cbc",
    "results": [
        {
            "text": "A CBC measures red blood cells...",
            "score": 0.664,
            "source": "cbc.md",
            "title": "Complete Blood Count (CBC) Guide",
            "chunk_id": "lab_tests/cbc::0",
            "document_id": "lab_tests/cbc",
            "section": "What a CBC measures",
            "category": "lab_tests",
        }
    ],
    "retrieval_count": 1,
    "top_k": 5,
    "score_threshold": 0.25,
    "timings": {"embed_ms": 9.0, "search_ms": 2.0, "total_ms": 11.0},
}


@pytest.fixture(name="ai_client")
def ai_client_fixture(client):
    """Give each test its own client instance, not the process singleton."""
    fresh = AIServiceClient()
    client.app.dependency_overrides[get_ai_client] = lambda: fresh
    return client


def error_body(code: str, message: str = "boom") -> dict:
    return {"error": {"code": code, "message": message}}


# ------------------------------------------------------------------ chat ---
@respx.mock
def test_chat_returns_answer_and_sources(ai_client):
    respx.post(f"{AI_BASE}/rag/query").mock(
        return_value=httpx.Response(200, json=ANSWER_PAYLOAD)
    )

    response = ai_client.post("/api/ai/chat", json={"question": "What is a CBC?"})

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["sources"][0]["source"] == "cbc.md"
    assert "not a medical diagnosis" in body["disclaimer"].lower()


@respx.mock
def test_chat_forwards_options(ai_client):
    route = respx.post(f"{AI_BASE}/rag/query").mock(
        return_value=httpx.Response(200, json=ANSWER_PAYLOAD)
    )

    ai_client.post(
        "/api/ai/chat", json={"question": "cbc", "top_k": 3, "category": "faq"}
    )

    import json

    sent = json.loads(route.calls[0].request.content)
    assert sent == {"question": "cbc", "top_k": 3, "category": "faq"}


@respx.mock
def test_unset_options_are_omitted_so_defaults_apply(ai_client):
    route = respx.post(f"{AI_BASE}/rag/query").mock(
        return_value=httpx.Response(200, json=ANSWER_PAYLOAD)
    )

    ai_client.post("/api/ai/chat", json={"question": "cbc"})

    import json

    assert json.loads(route.calls[0].request.content) == {"question": "cbc"}


@respx.mock
def test_request_id_is_propagated_to_the_ai_service(ai_client):
    route = respx.post(f"{AI_BASE}/rag/query").mock(
        return_value=httpx.Response(200, json=ANSWER_PAYLOAD)
    )

    ai_client.post(
        "/api/ai/chat",
        json={"question": "cbc"},
        headers={"X-Request-ID": "trace-across-services"},
    )

    assert route.calls[0].request.headers["X-Request-ID"] == "trace-across-services"


@respx.mock
def test_ungrounded_answer_is_passed_through_faithfully(ai_client):
    """A refusal must reach the UI as a refusal, not be dressed up."""
    payload = {
        **ANSWER_PAYLOAD,
        "answer": "I don't have enough information...",
        "sources": [],
        "retrieval_count": 0,
        "grounded": False,
        "finish_reason": "no_context",
    }
    respx.post(f"{AI_BASE}/rag/query").mock(return_value=httpx.Response(200, json=payload))

    body = ai_client.post("/api/ai/chat", json={"question": "football"}).json()

    assert body["grounded"] is False
    assert body["sources"] == []


# ---------------------------------------------------------------- search ---
@respx.mock
def test_search_returns_ranked_hits(ai_client):
    respx.post(f"{AI_BASE}/rag/search").mock(
        return_value=httpx.Response(200, json=SEARCH_PAYLOAD)
    )

    body = ai_client.post("/api/ai/search", json={"query": "cbc"}).json()

    assert body["retrieval_count"] == 1
    assert body["results"][0]["title"].startswith("Complete Blood Count")


# ------------------------------------------------------------- failures ----
@respx.mock
def test_connection_refused_becomes_503(ai_client):
    respx.post(f"{AI_BASE}/rag/query").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    response = ai_client.post("/api/ai/chat", json={"question": "cbc"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "AI_SERVICE_UNAVAILABLE"
    assert "unaffected" in response.json()["error"]["message"]


@respx.mock
def test_read_timeout_becomes_504(ai_client):
    respx.post(f"{AI_BASE}/rag/query").mock(
        side_effect=httpx.ReadTimeout("too slow")
    )

    response = ai_client.post("/api/ai/chat", json={"question": "cbc"})

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "AI_SERVICE_TIMEOUT"


@respx.mock
def test_generation_disabled_code_is_forwarded(ai_client):
    """The browser must distinguish 'switched off' from 'broken'."""
    respx.post(f"{AI_BASE}/rag/query").mock(
        return_value=httpx.Response(503, json=error_body("GENERATION_DISABLED"))
    )

    response = ai_client.post("/api/ai/chat", json={"question": "cbc"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "GENERATION_DISABLED"


@respx.mock
def test_model_unavailable_code_is_forwarded(ai_client):
    respx.post(f"{AI_BASE}/rag/query").mock(
        return_value=httpx.Response(503, json=error_body("MODEL_UNAVAILABLE"))
    )

    assert (
        ai_client.post("/api/ai/chat", json={"question": "cbc"}).json()["error"]["code"]
        == "MODEL_UNAVAILABLE"
    )


@respx.mock
def test_vector_store_unavailable_code_is_forwarded(ai_client):
    respx.post(f"{AI_BASE}/rag/search").mock(
        return_value=httpx.Response(503, json=error_body("VECTOR_STORE_UNAVAILABLE"))
    )

    assert (
        ai_client.post("/api/ai/search", json={"query": "cbc"}).json()["error"]["code"]
        == "VECTOR_STORE_UNAVAILABLE"
    )


@respx.mock
def test_upstream_500_becomes_502(ai_client):
    respx.post(f"{AI_BASE}/rag/query").mock(
        return_value=httpx.Response(500, json=error_body("INTERNAL_ERROR"))
    )

    assert ai_client.post("/api/ai/chat", json={"question": "cbc"}).status_code == 502


@respx.mock
def test_upstream_422_status_is_preserved(ai_client):
    respx.post(f"{AI_BASE}/rag/query").mock(
        return_value=httpx.Response(422, json=error_body("EMPTY_QUERY"))
    )

    response = ai_client.post("/api/ai/chat", json={"question": "x"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "EMPTY_QUERY"


@respx.mock
def test_malformed_json_becomes_502(ai_client):
    respx.post(f"{AI_BASE}/rag/query").mock(
        return_value=httpx.Response(200, content=b"<html>not json</html>")
    )

    response = ai_client.post("/api/ai/chat", json={"question": "cbc"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "AI_SERVICE_ERROR"


@respx.mock
def test_non_object_payload_becomes_502(ai_client):
    respx.post(f"{AI_BASE}/rag/query").mock(
        return_value=httpx.Response(200, json=["unexpected"])
    )

    assert ai_client.post("/api/ai/chat", json={"question": "cbc"}).status_code == 502


@respx.mock
def test_no_traceback_leaks_to_the_client(ai_client):
    respx.post(f"{AI_BASE}/rag/query").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    body = ai_client.post("/api/ai/chat", json={"question": "cbc"}).text

    assert "Traceback" not in body
    assert "httpx" not in body


# ---------------------------------------------------------------- health ---
@respx.mock
def test_health_reports_reachable(ai_client):
    respx.get(f"{AI_BASE}/health").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "ok",
                "service": "ai-service",
                "components": [{"name": "embedding_model", "state": "ok"}],
            },
        )
    )

    body = ai_client.get("/api/ai/health").json()

    assert body["reachable"] is True
    assert body["status"] == "ok"
    assert body["components"][0]["name"] == "embedding_model"


@respx.mock
def test_health_reports_unreachable_without_failing(ai_client):
    """A down dependency is a fact to report, not a 500 from this API."""
    respx.get(f"{AI_BASE}/health").mock(side_effect=httpx.ConnectError("refused"))

    response = ai_client.get("/api/ai/health")

    assert response.status_code == 200
    assert response.json()["reachable"] is False
    assert response.json()["status"] == "unreachable"


@respx.mock
def test_health_reports_degraded(ai_client):
    respx.get(f"{AI_BASE}/health").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "degraded",
                "components": [
                    {
                        "name": "generation_model",
                        "state": "unavailable",
                        "detail": "not installed",
                    }
                ],
            },
        )
    )

    body = ai_client.get("/api/ai/health").json()

    assert body["reachable"] is True
    assert body["status"] == "degraded"


# ---------------------------------------------- reports stay independent ---
@respx.mock
def test_reports_still_work_while_ai_is_down(ai_client, report_payload):
    """Rule 19: an AI outage must not touch core report functionality."""
    respx.post(f"{AI_BASE}/rag/query").mock(
        side_effect=httpx.ConnectError("refused")
    )
    respx.get(f"{AI_BASE}/health").mock(side_effect=httpx.ConnectError("refused"))

    assert ai_client.post("/api/ai/chat", json={"question": "x"}).status_code == 503

    created = ai_client.post("/api/reports", json=report_payload)
    listed = ai_client.get("/api/reports")
    dashboard = ai_client.get("/api/dashboard")
    health = ai_client.get("/health")

    assert created.status_code == 201
    assert listed.status_code == 200
    assert dashboard.status_code == 200
    assert health.json()["status"] == "ok"


# ------------------------------------------------------------ validation ---
def test_blank_question_is_rejected_before_calling_the_ai_service(ai_client):
    response = ai_client.post("/api/ai/chat", json={"question": ""})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_overlong_question_is_rejected(ai_client):
    assert (
        ai_client.post("/api/ai/chat", json={"question": "a" * 1001}).status_code == 422
    )
