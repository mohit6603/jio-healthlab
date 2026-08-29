"""Dashboard aggregation tests."""

from datetime import UTC, datetime, timedelta

import pytest


@pytest.fixture(name="dashboard_data")
def dashboard_data_fixture(client):
    soon = (datetime.now(UTC) + timedelta(hours=3)).replace(tzinfo=None).isoformat()
    far = (datetime.now(UTC) + timedelta(days=5)).replace(tzinfo=None).isoformat()
    rows = [
        {
            "patient_name": "Asha Nair",
            "age": 30,
            "test_type": "CBC Panel",
            "city": "Mumbai",
            "status": "processing",
            "priority": "urgent",
            "result_due_at": soon,
        },
        {
            "patient_name": "Rohan Kulkarni",
            "age": 50,
            "test_type": "Thyroid",
            "city": "Pune",
            "status": "ready",
            "priority": "routine",
            "result_due_at": far,
        },
        {
            "patient_name": "Devika Menon",
            "age": 40,
            "test_type": "CBC Panel",
            "city": "Mumbai",
            "status": "registered",
            "priority": "urgent",
            "result_due_at": soon,
        },
    ]
    for row in rows:
        assert client.post("/api/reports", json=row).status_code == 201


def test_dashboard_totals(client, dashboard_data):
    body = client.get("/api/dashboard").json()

    assert body["total_reports"] == 3
    assert body["ready_reports"] == 1
    assert body["urgent_reports"] == 2
    assert body["unique_tests"] == 2
    assert body["avg_age"] == 40.0


def test_dashboard_groupings(client, dashboard_data):
    body = client.get("/api/dashboard").json()

    assert body["by_city"] == {"Mumbai": 2, "Pune": 1}
    assert body["by_test_type"] == {"CBC Panel": 2, "Thyroid": 1}
    assert body["by_status"]["processing"] == 1


def test_due_soon_excludes_completed_and_distant_reports(client, dashboard_data):
    body = client.get("/api/dashboard").json()

    due_ids = {row["test_type"] for row in body["due_soon"]}
    assert due_ids == {"CBC Panel"}
    assert len(body["due_soon"]) == 2


def test_dashboard_on_empty_database(client):
    body = client.get("/api/dashboard").json()

    assert body["total_reports"] == 0
    assert body["avg_age"] is None
    assert body["latest_report_id"] is None
    assert body["due_soon"] == []
    assert body["by_status"] == {}
