"""Metrics endpoint tests."""

import httpx
import pytest
import respx

from app.security import Role

AI_BASE = "http://ai-service:8001"

REPORT = {
    "patient_name": "Asha Nair",
    "age": 34,
    "test_type": "CBC Panel",
    "city": "Mumbai",
    "status": "processing",
    "priority": "urgent",
}


def parse(text: str) -> dict[str, float]:
    """Parse the exposition format into {series: value}."""
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        series, _, value = line.rpartition(" ")
        values[series] = float(value)
    return values


def test_metrics_are_prometheus_text(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# TYPE" in response.text
    assert "# HELP" in response.text


def test_build_info_reports_the_version(client):
    values = parse(client.get("/metrics").text)

    assert any(
        series.startswith("healthlab_build_info{version=") for series in values
    )


def test_report_counters_reflect_the_database(client):
    for _ in range(3):
        client.post("/api/reports", json=REPORT)

    values = parse(client.get("/metrics").text)

    assert values["healthlab_reports_total"] == 3
    assert values['healthlab_reports_by_status{status="processing"}'] == 3
    assert values['healthlab_reports_by_priority{priority="urgent"}'] == 3


def test_counters_are_zero_on_an_empty_database(client):
    values = parse(client.get("/metrics").text)

    assert values["healthlab_reports_total"] == 0


@respx.mock
def test_ai_usage_series_are_exported(client):
    respx.post(f"{AI_BASE}/rag/query").mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "A CBC measures blood cells.",
                "sources": [],
                "retrieval_count": 2,
                "grounded": True,
                "disclaimer": "Not a medical diagnosis.",
                "model": "m",
                "provider": "local",
                "finish_reason": "stop",
                "timings": {"retrieval_ms": 1, "generation_ms": 2, "total_ms": 3},
            },
        )
    )
    client.post("/api/ai/chat", json={"question": "cbc"})

    values = parse(client.get("/metrics").text)

    assert values['healthlab_ai_queries_total{query_type="chat"}'] == 1
    assert values['healthlab_ai_query_failures_total{query_type="chat"}'] == 0


@respx.mock
def test_ai_failures_are_counted(client):
    respx.post(f"{AI_BASE}/rag/query").mock(
        side_effect=httpx.ConnectError("refused")
    )
    client.post("/api/ai/chat", json={"question": "cbc"})

    values = parse(client.get("/metrics").text)

    assert values['healthlab_ai_query_failures_total{query_type="chat"}'] == 1


def test_audit_events_are_exported(client):
    client.post("/api/reports", json=REPORT)

    values = parse(client.get("/metrics").text)

    assert (
        values['healthlab_audit_events_total{action="report.created",status="success"}']
        == 1
    )


def test_label_values_are_escaped():
    from app.services.metrics_service import MetricWriter

    writer = MetricWriter()
    writer.metric("x", 1, labels={"k": 'a"b\\c'})

    assert 'k="a\\"b\\\\c"' in writer.render()


def test_type_and_help_are_declared_once_per_series(client):
    for status in ("processing", "ready"):
        client.post("/api/reports", json={**REPORT, "status": status})

    text = client.get("/metrics").text

    assert text.count("# TYPE healthlab_reports_by_status") == 1
    assert text.count("# HELP healthlab_reports_by_status") == 1


@pytest.mark.parametrize("role", [Role.LAB_TECH, Role.DOCTOR, Role.VIEWER])
def test_metrics_are_admin_only(as_role, role):
    assert as_role(role).get("/metrics").status_code == 403


def test_metrics_require_authentication(raw_client):
    assert raw_client.get("/metrics").status_code == 401
