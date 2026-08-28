"""Report endpoint behaviour: filtering, validation and error contract."""

import pytest


@pytest.fixture(name="seeded")
def seeded_fixture(client):
    """Insert a small, deterministic set of reports."""
    rows = [
        {
            "patient_name": "Asha Nair",
            "age": 34,
            "test_type": "CBC Panel",
            "city": "Mumbai",
            "lab_branch": "BKC Flagship",
            "status": "processing",
            "priority": "urgent",
        },
        {
            "patient_name": "Rohan Kulkarni",
            "age": 52,
            "test_type": "Thyroid",
            "city": "Pune",
            "lab_branch": "Koregaon Park",
            "status": "ready",
            "priority": "routine",
        },
        {
            "patient_name": "Devika Menon",
            "age": 41,
            "test_type": "Kidney Function",
            "city": "Mumbai",
            "lab_branch": "Andheri Hub",
            "status": "registered",
            "priority": "urgent",
        },
    ]
    created = [client.post("/api/reports", json=row).json() for row in rows]
    return created


def test_list_returns_all_reports(client, seeded):
    response = client.get("/api/reports")

    assert response.status_code == 200
    assert len(response.json()) == 3


def test_filter_by_city(client, seeded):
    response = client.get("/api/reports", params={"city": "Mumbai"})

    assert response.status_code == 200
    assert {row["city"] for row in response.json()} == {"Mumbai"}


def test_filter_by_priority(client, seeded):
    response = client.get("/api/reports", params={"priority": "urgent"})

    assert len(response.json()) == 2


def test_filter_sentinel_all_is_ignored(client, seeded):
    response = client.get(
        "/api/reports", params={"city": "all", "status": "all", "priority": "all"}
    )

    assert len(response.json()) == 3


def test_search_matches_test_type(client, seeded):
    response = client.get("/api/reports", params={"search": "Thyroid"})

    assert len(response.json()) == 1
    assert response.json()[0]["test_type"] == "Thyroid"


def test_search_by_numeric_id(client, seeded):
    target = seeded[0]
    response = client.get("/api/reports", params={"search": str(target["id"])})

    assert [row["id"] for row in response.json()] == [target["id"]]


def test_pagination_limit_and_offset(client, seeded):
    page = client.get("/api/reports", params={"limit": 2, "offset": 0}).json()
    rest = client.get("/api/reports", params={"limit": 2, "offset": 2}).json()

    assert len(page) == 2
    assert len(rest) == 1
    assert {row["id"] for row in page}.isdisjoint({row["id"] for row in rest})


def test_get_single_report(client, seeded):
    target = seeded[0]
    response = client.get(f"/api/reports/{target['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == target["id"]


def test_missing_report_uses_error_envelope(client):
    response = client.get("/api/reports/999999")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "REPORT_NOT_FOUND"
    assert body["error"]["message"] == "Report not found"


def test_update_missing_report_returns_404(client):
    response = client.put("/api/reports/999999", json={"status": "ready"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "REPORT_NOT_FOUND"


def test_delete_missing_report_returns_404(client):
    response = client.delete("/api/reports/999999")

    assert response.status_code == 404


def test_partial_update_leaves_other_fields_intact(client, seeded):
    target = seeded[1]
    response = client.put(f"/api/reports/{target['id']}", json={"status": "delivered"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "delivered"
    assert body["patient_name"] == target["patient_name"]
    assert body["test_type"] == target["test_type"]


def test_invalid_payload_returns_validation_envelope(client):
    response = client.post(
        "/api/reports",
        json={"patient_name": "X", "age": 900, "test_type": "CBC Panel"},
    )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert {field["field"] for field in error["details"]["fields"]} == {
        "patient_name",
        "age",
    }


def test_invalid_status_value_rejected(client, report_payload):
    response = client.post(
        "/api/reports", json={**report_payload, "status": "teleported"}
    )

    assert response.status_code == 422


def test_blank_strings_are_stored_as_null(client, report_payload):
    response = client.post(
        "/api/reports", json={**report_payload, "doctor_name": "   ", "notes": ""}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["doctor_name"] is None
    assert body["notes"] is None


def test_limit_bounds_are_enforced(client):
    assert client.get("/api/reports", params={"limit": 0}).status_code == 422
    assert client.get("/api/reports", params={"limit": 501}).status_code == 422
    assert client.get("/api/reports", params={"offset": -1}).status_code == 422
