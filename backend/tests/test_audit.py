"""Audit trail tests.

Two things matter: the right actions get recorded, and nothing sensitive does.
"""

import httpx
import pytest
import respx

from app.models import AIQueryLog, AuditLog
from app.security import Role
from app.services import audit_service
from app.services.audit_service import Action, QueryType

AI_BASE = "http://ai-service:8001"

REPORT = {
    "patient_name": "Asha Nair",
    "age": 34,
    "test_type": "CBC Panel",
    "city": "Mumbai",
    "lab_branch": "Andheri Hub",
    "status": "processing",
    "priority": "urgent",
    "notes": "Call 9876543210. Sister is Priya.",
}

ANSWER = {
    "answer": "A CBC measures blood cells.",
    "sources": [{"title": "T", "source": "cbc.md", "chunk_id": "c::0", "score": 0.6}],
    "retrieval_count": 1,
    "grounded": True,
    "disclaimer": "Not a medical diagnosis.",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "provider": "local",
    "finish_reason": "stop",
    "timings": {"retrieval_ms": 1, "generation_ms": 2, "total_ms": 3},
}


def entries(db, action=None) -> list[AuditLog]:
    return audit_service.list_audit_logs(db, action=action, limit=500)


def ai_entries(db) -> list[AIQueryLog]:
    return audit_service.list_ai_query_logs(db, limit=500)


# ------------------------------------------------------------------ auth ---
def test_successful_login_is_recorded(client, db):
    recorded = entries(db, str(Action.LOGIN_SUCCEEDED))

    assert recorded
    assert recorded[0].status == "success"
    assert recorded[0].user_role == "ADMIN"


def test_failed_login_is_recorded(raw_client, db):
    raw_client.post(
        "/api/auth/login",
        json={"email": "nobody@test.example.com", "password": "wrong"},
    )

    recorded = entries(db, str(Action.LOGIN_FAILED))
    assert recorded
    assert recorded[0].status == "failure"
    assert recorded[0].detail == "INVALID_CREDENTIALS"


def test_failed_login_does_not_store_the_attempted_email(raw_client, db):
    """The address is unverified input; recording it fills the trail with
    whatever anyone types into the form."""
    raw_client.post(
        "/api/auth/login",
        json={"email": "someone.private@test.example.com", "password": "wrong"},
    )

    assert "someone.private" not in str(
        [entry.detail for entry in entries(db, str(Action.LOGIN_FAILED))]
    )


def test_logout_is_recorded(client, db):
    session = client.post(
        "/api/auth/login",
        json={"email": "admin@test.example.com", "password": "TestPassword!2026"},
    ).json()

    client.post("/api/auth/logout", json={"refresh_token": session["refresh_token"]})

    assert entries(db, str(Action.LOGOUT))


def test_refresh_token_reuse_is_recorded(client, db):
    session = client.post(
        "/api/auth/login",
        json={"email": "admin@test.example.com", "password": "TestPassword!2026"},
    ).json()
    client.post("/api/auth/refresh", json={"refresh_token": session["refresh_token"]})

    client.post("/api/auth/refresh", json={"refresh_token": session["refresh_token"]})

    assert entries(db, str(Action.SESSION_REUSE_DETECTED))


def test_user_creation_is_recorded(client, db):
    client.post(
        "/api/auth/users",
        json={
            "email": "audited@test.example.com",
            "full_name": "Audited User",
            "password": "AnotherStrong!2026",
            "role": "LAB_TECH",
        },
    )

    recorded = entries(db, str(Action.USER_CREATED))
    assert recorded
    assert "role=LAB_TECH" in recorded[0].detail


# --------------------------------------------------------------- reports ---
def test_report_creation_is_recorded(client, db):
    report_id = client.post("/api/reports", json=REPORT).json()["id"]

    recorded = entries(db, str(Action.REPORT_CREATED))
    assert recorded
    assert recorded[0].resource_id == str(report_id)
    assert recorded[0].resource_type == "report"


def test_report_update_records_field_names_only(client, db):
    """Values could contain a patient name or a free-text note."""
    report_id = client.post("/api/reports", json=REPORT).json()["id"]

    client.put(
        f"/api/reports/{report_id}",
        json={"status": "ready", "patient_name": "Renamed Patient"},
    )

    recorded = entries(db, str(Action.REPORT_UPDATED))[0]
    assert "fields=" in recorded.detail
    assert "patient_name" in recorded.detail
    assert "Renamed Patient" not in recorded.detail


def test_report_deletion_is_recorded(client, db):
    report_id = client.post("/api/reports", json=REPORT).json()["id"]

    client.delete(f"/api/reports/{report_id}")

    assert entries(db, str(Action.REPORT_DELETED))[0].resource_id == str(report_id)


def test_no_patient_identifier_reaches_the_audit_trail(client, db):
    report_id = client.post("/api/reports", json=REPORT).json()["id"]
    client.put(f"/api/reports/{report_id}", json={"status": "ready"})
    client.delete(f"/api/reports/{report_id}")

    blob = " ".join(
        f"{entry.detail or ''} {entry.resource_id or ''}" for entry in entries(db)
    )
    for identifier in ("Asha Nair", "9876543210", "Priya"):
        assert identifier not in blob


# -------------------------------------------------------------------- AI ---
@respx.mock
def test_ai_chat_records_metadata_but_not_the_question(client, db):
    respx.post(f"{AI_BASE}/rag/query").mock(
        return_value=httpx.Response(200, json=ANSWER)
    )

    client.post(
        "/api/ai/chat",
        json={"question": "Does patient Asha Nair have anaemia?"},
    )

    recorded = ai_entries(db)
    assert recorded
    entry = recorded[0]
    assert entry.query_type == str(QueryType.CHAT)
    assert entry.success is True
    assert entry.source_count == 1
    assert entry.grounded is True
    assert entry.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert entry.latency_ms is not None

    # The question must not be recoverable from any column.
    blob = " ".join(str(value) for value in vars(entry).values())
    assert "Asha Nair" not in blob
    assert "anaemia" not in blob


@respx.mock
def test_ai_failure_records_the_error_code(client, db):
    respx.post(f"{AI_BASE}/rag/query").mock(
        side_effect=httpx.ConnectError("refused")
    )

    client.post("/api/ai/chat", json={"question": "cbc"})

    entry = ai_entries(db)[0]
    assert entry.success is False
    assert entry.error_code == "AI_SERVICE_UNAVAILABLE"


@respx.mock
def test_report_explanation_is_recorded_twice(client, db):
    """Once as an action on a report, once as AI usage."""
    respx.post(f"{AI_BASE}/rag/explain").mock(
        return_value=httpx.Response(200, json=ANSWER)
    )
    report_id = client.post("/api/reports", json=REPORT).json()["id"]

    client.post(f"/api/reports/{report_id}/explain")

    action = entries(db, str(Action.AI_REPORT_EXPLAINED))[0]
    assert action.resource_id == str(report_id)
    assert action.latency_ms is not None
    assert any(item.query_type == str(QueryType.EXPLAIN) for item in ai_entries(db))


@respx.mock
def test_index_rebuild_is_recorded(client, db):
    respx.post(f"{AI_BASE}/reports/index").mock(
        return_value=httpx.Response(201, json={"indexed": 3, "collection": "c"})
    )
    client.post("/api/reports", json=REPORT)

    client.post("/api/ai/report-index")

    assert "indexed=3" in entries(db, str(Action.REPORT_INDEX_REBUILT))[0].detail


# ----------------------------------------------------------- correlation ---
def test_request_id_is_recorded_for_correlation(client, db):
    client.post("/api/reports", json=REPORT, headers={"X-Request-ID": "audit-trace"})

    assert entries(db, str(Action.REPORT_CREATED))[0].request_id == "audit-trace"


# --------------------------------------------------------------- service ---
def test_audit_write_failure_does_not_raise(db, monkeypatch):
    """An unwritable trail must not turn a successful action into a 500."""

    def explode():
        raise RuntimeError("disk full")

    monkeypatch.setattr(db, "commit", explode)

    assert audit_service.record(db, action=Action.REPORT_CREATED) is None


def test_ai_usage_summary_groups_by_type(client, db):
    for query_type in (QueryType.CHAT, QueryType.CHAT, QueryType.SEARCH):
        audit_service.record_ai_query(db, query_type=query_type, latency_ms=10.0)
    audit_service.record_ai_query(
        db, query_type=QueryType.CHAT, success=False, error_code="X", latency_ms=5.0
    )

    summary = audit_service.ai_usage_summary(db)
    by_type = {row["query_type"]: row for row in summary["by_type"]}

    assert by_type["chat"]["count"] == 3
    assert by_type["chat"]["failures"] == 1
    assert by_type["search"]["count"] == 1


# ------------------------------------------------------------- endpoints ---
def test_audit_endpoint_returns_entries(client):
    client.post("/api/reports", json=REPORT)

    response = client.get("/api/audit")

    assert response.status_code == 200
    assert any(row["action"] == "report.created" for row in response.json())


def test_audit_endpoint_filters_by_action(client):
    client.post("/api/reports", json=REPORT)

    rows = client.get("/api/audit", params={"action": "report.created"}).json()

    assert rows
    assert all(row["action"] == "report.created" for row in rows)


def test_audit_endpoint_is_ordered_newest_first(client):
    for _ in range(3):
        client.post("/api/reports", json=REPORT)

    rows = client.get("/api/audit", params={"action": "report.created"}).json()

    assert [row["id"] for row in rows] == sorted(
        (row["id"] for row in rows), reverse=True
    )


@respx.mock
def test_ai_query_endpoint(client):
    respx.post(f"{AI_BASE}/rag/query").mock(
        return_value=httpx.Response(200, json=ANSWER)
    )
    client.post("/api/ai/chat", json={"question": "cbc"})

    rows = client.get("/api/audit/ai-queries").json()

    assert rows
    assert rows[0]["query_type"] == "chat"
    # The schema has no field that could hold a question.
    assert "question" not in rows[0]
    assert "answer" not in rows[0]


def test_ai_usage_endpoint(client):
    response = client.get("/api/audit/ai-usage")

    assert response.status_code == 200
    assert "by_type" in response.json()


@pytest.mark.parametrize("role", [Role.LAB_TECH, Role.DOCTOR, Role.VIEWER])
def test_audit_trail_is_admin_only(as_role, role):
    non_admin = as_role(role)

    assert non_admin.get("/api/audit").status_code == 403
    assert non_admin.get("/api/audit/ai-queries").status_code == 403
    assert non_admin.get("/api/audit/ai-usage").status_code == 403


def test_there_is_no_way_to_modify_the_trail(client):
    """Append-only: no update or delete route exists."""
    assert client.delete("/api/audit").status_code in {404, 405}
    assert client.put("/api/audit", json={}).status_code in {404, 405}


def test_audit_pagination(client):
    for _ in range(5):
        client.post("/api/reports", json=REPORT)

    page = client.get("/api/audit", params={"limit": 2}).json()

    assert len(page) == 2


def test_audit_limit_bounds(client):
    assert client.get("/api/audit", params={"limit": 0}).status_code == 422
    assert client.get("/api/audit", params={"limit": 501}).status_code == 422
