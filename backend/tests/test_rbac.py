"""RBAC tests.

The permission table is the contract; these tests assert it holds both ways --
every role can do what it should, and cannot do what it should not.
"""

import httpx
import pytest
import respx

from app.security import Permission, Role, has_permission, permissions_for

AI_BASE = "http://ai-service:8001"

REPORT = {
    "patient_name": "Asha Nair",
    "age": 34,
    "test_type": "CBC Panel",
    "city": "Mumbai",
    "lab_branch": "Andheri Hub",
    "status": "processing",
    "priority": "urgent",
}


# ------------------------------------------------------- permission table --
def test_admin_has_every_permission():
    assert permissions_for(Role.ADMIN) == frozenset(Permission)


def test_viewer_is_read_only():
    viewer = permissions_for(Role.VIEWER)

    assert Permission.REPORTS_READ in viewer
    assert Permission.REPORTS_CREATE not in viewer
    assert Permission.REPORTS_UPDATE not in viewer
    assert Permission.REPORTS_DELETE not in viewer


def test_viewer_gets_informational_ai_only():
    viewer = permissions_for(Role.VIEWER)

    assert Permission.AI_CHAT in viewer
    assert Permission.AI_SEARCH in viewer
    assert Permission.AI_EXPLAIN_REPORT not in viewer
    assert Permission.AI_RISK_ANALYTICS not in viewer


def test_lab_tech_can_move_work_through_the_pipeline():
    tech = permissions_for(Role.LAB_TECH)

    assert Permission.REPORTS_UPDATE in tech
    assert Permission.REPORTS_CREATE in tech
    assert Permission.AI_RISK_ANALYTICS in tech


def test_lab_tech_cannot_administer():
    tech = permissions_for(Role.LAB_TECH)

    assert Permission.ADMIN_USERS not in tech
    assert Permission.ADMIN_DOCUMENTS not in tech
    assert Permission.REPORTS_DELETE not in tech


def test_doctor_can_explain_reports_but_not_run_the_lab():
    doctor = permissions_for(Role.DOCTOR)

    assert Permission.AI_EXPLAIN_REPORT in doctor
    assert Permission.REPORTS_UPDATE not in doctor
    assert Permission.ADMIN_USERS not in doctor


def test_every_role_inherits_the_viewer_baseline():
    baseline = permissions_for(Role.VIEWER)

    for role in Role:
        assert baseline <= permissions_for(role), role


def test_unknown_role_grants_nothing():
    assert permissions_for("SUPERUSER") == frozenset()
    assert has_permission("SUPERUSER", Permission.REPORTS_READ) is False


# ----------------------------------------------------- enforcement: reads --
@pytest.mark.parametrize("role", list(Role))
def test_every_role_can_read_reports(as_role, role):
    client = as_role(role)

    assert client.get("/api/reports").status_code == 200


@pytest.mark.parametrize("role", list(Role))
def test_every_role_can_read_the_dashboard(as_role, role):
    client = as_role(role)

    assert client.get("/api/dashboard").status_code == 200


# ---------------------------------------------------- enforcement: writes --
@pytest.mark.parametrize(
    ("role", "expected"),
    [(Role.ADMIN, 201), (Role.LAB_TECH, 201), (Role.DOCTOR, 403), (Role.VIEWER, 403)],
)
def test_report_creation_is_restricted(as_role, role, expected):
    client = as_role(role)

    assert client.post("/api/reports", json=REPORT).status_code == expected


@pytest.mark.parametrize(
    ("role", "expected"),
    [(Role.ADMIN, 200), (Role.LAB_TECH, 200), (Role.DOCTOR, 403), (Role.VIEWER, 403)],
)
def test_report_updates_are_restricted(as_role, role, expected):
    admin = as_role(Role.ADMIN)
    report_id = admin.post("/api/reports", json=REPORT).json()["id"]
    client = as_role(role)

    response = client.put(f"/api/reports/{report_id}", json={"status": "ready"})

    assert response.status_code == expected


@pytest.mark.parametrize(
    ("role", "expected"),
    [(Role.ADMIN, 204), (Role.LAB_TECH, 403), (Role.DOCTOR, 403), (Role.VIEWER, 403)],
)
def test_report_deletion_is_admin_only(as_role, role, expected):
    admin = as_role(Role.ADMIN)
    report_id = admin.post("/api/reports", json=REPORT).json()["id"]
    client = as_role(role)

    assert client.delete(f"/api/reports/{report_id}").status_code == expected


# -------------------------------------------------------- enforcement: AI --
@respx.mock
@pytest.mark.parametrize("role", list(Role))
def test_every_role_may_use_the_knowledge_assistant(as_role, role):
    respx.post(f"{AI_BASE}/rag/query").mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "A CBC measures blood cells.",
                "sources": [],
                "retrieval_count": 0,
                "grounded": True,
                "disclaimer": "Not a medical diagnosis.",
                "model": "m",
                "provider": "p",
                "finish_reason": "stop",
                "timings": {"retrieval_ms": 1, "generation_ms": 1, "total_ms": 2},
            },
        )
    )
    client = as_role(role)

    assert client.post("/api/ai/chat", json={"question": "cbc"}).status_code == 200


@respx.mock
@pytest.mark.parametrize(
    ("role", "expected"),
    [(Role.ADMIN, 200), (Role.DOCTOR, 200), (Role.LAB_TECH, 403), (Role.VIEWER, 403)],
)
def test_report_explanation_is_restricted(as_role, role, expected):
    respx.post(f"{AI_BASE}/rag/explain").mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "A CBC measures blood cells.",
                "sources": [],
                "retrieval_count": 0,
                "grounded": True,
                "disclaimer": "Not a medical diagnosis.",
                "model": "m",
                "provider": "p",
                "finish_reason": "stop",
                "timings": {"retrieval_ms": 1, "generation_ms": 1, "total_ms": 2},
            },
        )
    )
    admin = as_role(Role.ADMIN)
    report_id = admin.post("/api/reports", json=REPORT).json()["id"]
    client = as_role(role)

    response = client.post(f"/api/reports/{report_id}/explain")

    assert response.status_code == expected


@respx.mock
@pytest.mark.parametrize(
    ("role", "expected"),
    [(Role.ADMIN, 200), (Role.LAB_TECH, 200), (Role.DOCTOR, 403), (Role.VIEWER, 403)],
)
def test_risk_analytics_is_operational_only(as_role, role, expected):
    respx.post(f"{AI_BASE}/ml/predict-delay/batch").mock(
        return_value=httpx.Response(
            200, json={"predictions": [], "model_version": "v1", "count": 0}
        )
    )
    client = as_role(role)

    assert client.get("/api/ai/risk-analytics").status_code == expected


@respx.mock
@pytest.mark.parametrize(
    ("role", "expected"),
    [(Role.ADMIN, 200), (Role.LAB_TECH, 200), (Role.DOCTOR, 200), (Role.VIEWER, 403)],
)
def test_report_search_excludes_viewers(as_role, role, expected):
    respx.post(f"{AI_BASE}/reports/search").mock(
        return_value=httpx.Response(
            200, json={"query": "q", "results": [], "retrieval_count": 0}
        )
    )
    client = as_role(role)

    response = client.post("/api/ai/report-search", json={"query": "kidney"})

    assert response.status_code == expected


@respx.mock
@pytest.mark.parametrize(
    ("role", "expected"),
    [(Role.ADMIN, 200), (Role.LAB_TECH, 403), (Role.DOCTOR, 403), (Role.VIEWER, 403)],
)
def test_rebuilding_the_report_index_is_admin_only(as_role, role, expected):
    respx.post(f"{AI_BASE}/reports/index").mock(
        return_value=httpx.Response(201, json={"indexed": 0, "collection": "c"})
    )
    client = as_role(role)

    assert client.post("/api/ai/report-index").status_code == expected


# ---------------------------------------------------- enforcement: admin ---
@pytest.mark.parametrize(
    ("role", "expected"),
    [(Role.ADMIN, 200), (Role.LAB_TECH, 403), (Role.DOCTOR, 403), (Role.VIEWER, 403)],
)
def test_user_administration_is_admin_only(as_role, role, expected):
    client = as_role(role)

    assert client.get("/api/auth/users").status_code == expected


def test_admin_can_create_users(as_role):
    client = as_role(Role.ADMIN)

    response = client.post(
        "/api/auth/users",
        json={
            "email": "new.tech@test.example.com",
            "full_name": "New Technician",
            "password": "AnotherStrong!2026",
            "role": "LAB_TECH",
        },
    )

    assert response.status_code == 201
    assert response.json()["role"] == "LAB_TECH"
    assert "password" not in response.json()


def test_duplicate_email_is_rejected(as_role):
    client = as_role(Role.ADMIN)
    payload = {
        "email": "dupe@test.example.com",
        "full_name": "Dupe",
        "password": "AnotherStrong!2026",
        "role": "VIEWER",
    }
    client.post("/api/auth/users", json=payload)

    response = client.post("/api/auth/users", json=payload)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "EMAIL_ALREADY_REGISTERED"


def test_short_passwords_are_rejected(as_role):
    client = as_role(Role.ADMIN)

    response = client.post(
        "/api/auth/users",
        json={
            "email": "weak@test.example.com",
            "full_name": "Weak",
            "password": "short",
            "role": "VIEWER",
        },
    )

    assert response.status_code == 422


# ------------------------------------------------------------- messaging ---
def test_denial_names_the_role_without_leaking_the_matrix(as_role):
    client = as_role(Role.VIEWER)

    body = client.post("/api/reports", json=REPORT).json()

    assert body["error"]["code"] == "PERMISSION_DENIED"
    assert "VIEWER" in body["error"]["message"]


def test_me_reports_the_permissions_the_server_enforces(as_role):
    for role in Role:
        client = as_role(role)
        reported = set(client.get("/api/auth/me").json()["permissions"])

        assert reported == {str(item) for item in permissions_for(role)}, role
