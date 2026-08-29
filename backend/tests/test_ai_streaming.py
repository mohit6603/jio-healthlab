"""Streaming proxy tests.

The backend must pass SSE frames through untouched and keep the AI-usage log
honest even when the client disconnects mid-answer.
"""

import httpx
import pytest
import respx

from app.security import Role
from app.services.ai_client import AIServiceClient, get_ai_client

AI_BASE = "http://ai-service:8001"

SSE_BODY = (
    'event: sources\ndata: {"sources": [{"title": "CBC Guide", "source": '
    '"cbc.md", "chunk_id": "c::0", "score": 0.66}]}\n\n'
    'event: token\ndata: {"text": "A CBC "}\n\n'
    'event: token\ndata: {"text": "measures blood cells."}\n\n'
    'event: done\ndata: {"grounded": true, "finish_reason": "stop", "model": '
    '"m", "provider": "local", "retrieval_count": 1, "drop_sources": false, '
    '"disclaimer": "Not a medical diagnosis.", "timings": {"retrieval_ms": 1, '
    '"generation_ms": 0, "total_ms": 2}}\n\n'
)


@pytest.fixture(name="stream_client")
def stream_client_fixture(client):
    client.app.dependency_overrides[get_ai_client] = lambda: AIServiceClient()
    return client


@respx.mock
def test_frames_are_passed_through_unchanged(stream_client):
    respx.post(f"{AI_BASE}/rag/query/stream").mock(
        return_value=httpx.Response(
            200, text=SSE_BODY, headers={"content-type": "text/event-stream"}
        )
    )

    response = stream_client.post("/api/ai/chat/stream", json={"question": "cbc"})

    assert response.status_code == 200
    assert "event: sources" in response.text
    assert "event: token" in response.text
    assert "event: done" in response.text
    assert "measures blood cells." in response.text


@respx.mock
def test_streaming_headers_prevent_buffering(stream_client):
    respx.post(f"{AI_BASE}/rag/query/stream").mock(
        return_value=httpx.Response(200, text=SSE_BODY)
    )

    response = stream_client.post("/api/ai/chat/stream", json={"question": "cbc"})

    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"


@respx.mock
def test_request_id_is_propagated(stream_client):
    route = respx.post(f"{AI_BASE}/rag/query/stream").mock(
        return_value=httpx.Response(200, text=SSE_BODY)
    )

    stream_client.post(
        "/api/ai/chat/stream",
        json={"question": "cbc"},
        headers={"X-Request-ID": "stream-trace"},
    )

    assert route.calls[0].request.headers["X-Request-ID"] == "stream-trace"


@respx.mock
def test_options_are_forwarded(stream_client):
    import json

    route = respx.post(f"{AI_BASE}/rag/query/stream").mock(
        return_value=httpx.Response(200, text=SSE_BODY)
    )

    stream_client.post(
        "/api/ai/chat/stream", json={"question": "cbc", "top_k": 3, "category": "faq"}
    )

    sent = json.loads(route.calls[0].request.content)
    assert sent == {"question": "cbc", "top_k": 3, "category": "faq"}


@respx.mock
def test_outage_becomes_an_error_event(stream_client):
    """The response has already begun, so a status code is not available."""
    respx.post(f"{AI_BASE}/rag/query/stream").mock(
        side_effect=httpx.ConnectError("refused")
    )

    response = stream_client.post("/api/ai/chat/stream", json={"question": "cbc"})

    assert "event: error" in response.text
    assert "AI_SERVICE_UNAVAILABLE" in response.text


@respx.mock
def test_upstream_error_status_is_translated(stream_client):
    respx.post(f"{AI_BASE}/rag/query/stream").mock(
        return_value=httpx.Response(
            503, json={"error": {"code": "GENERATION_DISABLED", "message": "off"}}
        )
    )

    response = stream_client.post("/api/ai/chat/stream", json={"question": "cbc"})

    assert "event: error" in response.text
    assert "GENERATION_DISABLED" in response.text


@respx.mock
def test_usage_is_logged_even_though_nothing_is_returned_synchronously(
    stream_client, db
):
    from app.services import audit_service

    respx.post(f"{AI_BASE}/rag/query/stream").mock(
        return_value=httpx.Response(200, text=SSE_BODY)
    )

    stream_client.post("/api/ai/chat/stream", json={"question": "cbc"})

    entries = audit_service.list_ai_query_logs(db, limit=10)
    assert entries
    assert entries[0].query_type == "chat"


@respx.mock
def test_failed_stream_records_the_error_code(stream_client, db):
    from app.services import audit_service

    respx.post(f"{AI_BASE}/rag/query/stream").mock(
        side_effect=httpx.ConnectError("refused")
    )

    stream_client.post("/api/ai/chat/stream", json={"question": "cbc"})

    entry = audit_service.list_ai_query_logs(db, limit=10)[0]
    assert entry.success is False
    assert entry.error_code == "AI_SERVICE_UNAVAILABLE"


@respx.mock
def test_the_question_is_never_stored(stream_client, db):
    from app.services import audit_service

    respx.post(f"{AI_BASE}/rag/query/stream").mock(
        return_value=httpx.Response(200, text=SSE_BODY)
    )

    stream_client.post(
        "/api/ai/chat/stream",
        json={"question": "Does patient Asha Nair have anaemia?"},
    )

    entry = audit_service.list_ai_query_logs(db, limit=10)[0]
    assert "Asha Nair" not in " ".join(str(v) for v in vars(entry).values())


def test_streaming_requires_authentication(raw_client):
    response = raw_client.post("/api/ai/chat/stream", json={"question": "cbc"})

    assert response.status_code == 401


@respx.mock
@pytest.mark.parametrize("role", list(Role))
def test_every_role_may_stream(as_role, role):
    respx.post(f"{AI_BASE}/rag/query/stream").mock(
        return_value=httpx.Response(200, text=SSE_BODY)
    )
    client = as_role(role)
    client.app.dependency_overrides[get_ai_client] = lambda: AIServiceClient()

    response = client.post("/api/ai/chat/stream", json={"question": "cbc"})

    assert response.status_code == 200


def test_blank_question_is_rejected_before_streaming(stream_client):
    response = stream_client.post("/api/ai/chat/stream", json={"question": ""})

    assert response.status_code == 422
