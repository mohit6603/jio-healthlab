"""Risk analytics tests.

Queue features are derived from the database, so these tests seed real reports
and assert the derived numbers, not just the plumbing.
"""

import json

import httpx
import pytest
import respx

from app.services.ai_client import AIServiceClient, get_ai_client

AI_BASE = "http://ai-service:8001"


def prediction(probability: float, level: str) -> dict:
    return {
        "delay_probability": probability,
        "risk_level": level,
        "model_version": "v20260101000000",
        "model_name": "report-delay-classifier",
        "algorithm": "logistic_regression",
        "inferred_fields": {},
        "synthetic_model": True,
    }


def batch(*items: dict) -> dict:
    return {
        "predictions": list(items),
        "model_version": "v20260101000000",
        "count": len(items),
    }


@pytest.fixture(name="risk_client")
def risk_client_fixture(client):
    client.app.dependency_overrides[get_ai_client] = lambda: AIServiceClient()
    return client


@pytest.fixture(name="seeded")
def seeded_fixture(risk_client):
    """Three in-flight reports and two that can no longer be late."""
    rows = [
        {
            "patient_name": "Asha Nair",
            "age": 34,
            "test_type": "CBC Panel",
            "city": "Mumbai",
            "lab_branch": "Andheri Hub",
            "status": "processing",
            "priority": "urgent",
        },
        {
            "patient_name": "Rohan Kulkarni",
            "age": 52,
            "test_type": "Thyroid",
            "city": "Mumbai",
            "lab_branch": "Andheri Hub",
            "status": "registered",
            "priority": "routine",
        },
        {
            "patient_name": "Devika Menon",
            "age": 41,
            "test_type": "CBC Panel",
            "city": "Pune",
            "lab_branch": "Koregaon Park",
            "status": "collected",
            "priority": "routine",
        },
        {
            "patient_name": "Already Done",
            "age": 30,
            "test_type": "CBC Panel",
            "city": "Mumbai",
            "lab_branch": "Andheri Hub",
            "status": "ready",
            "priority": "urgent",
        },
        {
            "patient_name": "Also Done",
            "age": 30,
            "test_type": "Thyroid",
            "city": "Mumbai",
            "lab_branch": "Andheri Hub",
            "status": "delivered",
            "priority": "routine",
        },
    ]
    for row in rows:
        assert risk_client.post("/api/reports", json=row).status_code == 201


# ------------------------------------------------------------- selection ---
@respx.mock
def test_only_in_flight_reports_are_scored(risk_client, seeded):
    route = respx.post(f"{AI_BASE}/ml/predict-delay/batch").mock(
        return_value=httpx.Response(
            200,
            json=batch(
                prediction(0.9, "high"),
                prediction(0.5, "medium"),
                prediction(0.1, "low"),
            ),
        )
    )

    body = risk_client.get("/api/ai/risk-analytics").json()

    sent = json.loads(route.calls[0].request.content)["items"]
    assert len(sent) == 3
    assert body["reports_scored"] == 3


@respx.mock
def test_queue_features_come_from_the_database(risk_client, seeded):
    """Andheri Hub has 2 in-flight reports, 1 of them urgent."""
    route = respx.post(f"{AI_BASE}/ml/predict-delay/batch").mock(
        return_value=httpx.Response(
            200,
            json=batch(
                prediction(0.9, "high"),
                prediction(0.5, "medium"),
                prediction(0.1, "low"),
            ),
        )
    )

    risk_client.get("/api/ai/risk-analytics")

    sent = json.loads(route.calls[0].request.content)["items"]
    andheri = [item for item in sent if item.get("branch") == "Andheri Hub"]
    assert len(andheri) == 2
    assert all(item["queue_size"] == 2 for item in andheri)
    assert all(item["urgent_report_count"] == 1 for item in andheri)


@respx.mock
def test_no_patient_identifier_is_sent(risk_client, seeded):
    route = respx.post(f"{AI_BASE}/ml/predict-delay/batch").mock(
        return_value=httpx.Response(
            200,
            json=batch(
                prediction(0.9, "high"),
                prediction(0.5, "medium"),
                prediction(0.1, "low"),
            ),
        )
    )

    risk_client.get("/api/ai/risk-analytics")

    sent = route.calls[0].request.content.decode()
    for name in ("Asha Nair", "Rohan Kulkarni", "Devika Menon"):
        assert name not in sent


@respx.mock
def test_response_carries_no_patient_identifier(risk_client, seeded):
    respx.post(f"{AI_BASE}/ml/predict-delay/batch").mock(
        return_value=httpx.Response(
            200,
            json=batch(
                prediction(0.9, "high"),
                prediction(0.5, "medium"),
                prediction(0.1, "low"),
            ),
        )
    )

    body = risk_client.get("/api/ai/risk-analytics").text

    for name in ("Asha Nair", "Rohan Kulkarni", "Devika Menon"):
        assert name not in body


# ------------------------------------------------------------ aggregation --
@respx.mock
def test_summary_counters(risk_client, seeded):
    respx.post(f"{AI_BASE}/ml/predict-delay/batch").mock(
        return_value=httpx.Response(
            200,
            json=batch(
                prediction(0.90, "high"),
                prediction(0.50, "medium"),
                prediction(0.10, "low"),
            ),
        )
    )

    body = risk_client.get("/api/ai/risk-analytics").json()

    assert body["reports_scored"] == 3
    assert body["high_risk"] == 1
    assert body["at_risk"] == 2
    assert body["predicted_late"] == 2
    assert body["average_probability"] == pytest.approx(0.5, abs=1e-3)


@respx.mock
def test_risk_distribution(risk_client, seeded):
    respx.post(f"{AI_BASE}/ml/predict-delay/batch").mock(
        return_value=httpx.Response(
            200,
            json=batch(
                prediction(0.90, "high"),
                prediction(0.50, "medium"),
                prediction(0.10, "low"),
            ),
        )
    )

    body = risk_client.get("/api/ai/risk-analytics").json()

    assert body["risk_distribution"] == {"low": 1, "medium": 1, "high": 1}


@respx.mock
def test_reports_are_ordered_riskiest_first(risk_client, seeded):
    respx.post(f"{AI_BASE}/ml/predict-delay/batch").mock(
        return_value=httpx.Response(
            200,
            json=batch(
                prediction(0.20, "low"),
                prediction(0.95, "high"),
                prediction(0.55, "medium"),
            ),
        )
    )

    probabilities = [
        row["delay_probability"]
        for row in risk_client.get("/api/ai/risk-analytics").json()["reports"]
    ]

    assert probabilities == sorted(probabilities, reverse=True)


@respx.mock
def test_highest_risk_branch_is_identified(risk_client, seeded):
    respx.post(f"{AI_BASE}/ml/predict-delay/batch").mock(
        return_value=httpx.Response(
            200,
            json=batch(
                prediction(0.90, "high"),
                prediction(0.80, "high"),
                prediction(0.05, "low"),
            ),
        )
    )

    body = risk_client.get("/api/ai/risk-analytics").json()

    # Order-independent: whichever branch tops by_branch must be the one
    # reported, and by_branch must itself be sorted worst-first.
    averages = [row["average_probability"] for row in body["by_branch"]]
    assert averages == sorted(averages, reverse=True)
    assert body["highest_risk_branch"]["branch"] == body["by_branch"][0]["label"]
    assert body["highest_risk_branch"]["average_probability"] == max(averages)


@respx.mock
def test_grouping_by_branch_and_test_type(risk_client, seeded):
    respx.post(f"{AI_BASE}/ml/predict-delay/batch").mock(
        return_value=httpx.Response(
            200,
            json=batch(
                prediction(0.90, "high"),
                prediction(0.80, "high"),
                prediction(0.10, "low"),
            ),
        )
    )

    body = risk_client.get("/api/ai/risk-analytics").json()

    branches = {row["label"]: row for row in body["by_branch"]}
    assert branches["Andheri Hub"]["count"] == 2
    assert branches["Koregaon Park"]["count"] == 1
    assert {row["label"] for row in body["by_test_type"]} == {"CBC Panel", "Thyroid"}


@respx.mock
def test_model_version_is_reported(risk_client, seeded):
    respx.post(f"{AI_BASE}/ml/predict-delay/batch").mock(
        return_value=httpx.Response(
            200,
            json=batch(
                prediction(0.5, "medium"),
                prediction(0.5, "medium"),
                prediction(0.5, "medium"),
            ),
        )
    )

    body = risk_client.get("/api/ai/risk-analytics").json()

    assert body["model_version"] == "v20260101000000"
    assert body["synthetic_model"] is True


@respx.mock
def test_limit_is_respected(risk_client, seeded):
    route = respx.post(f"{AI_BASE}/ml/predict-delay/batch").mock(
        return_value=httpx.Response(200, json=batch(prediction(0.5, "medium")))
    )

    risk_client.get("/api/ai/risk-analytics", params={"limit": 1})

    assert len(json.loads(route.calls[0].request.content)["items"]) == 1


# ------------------------------------------------------------ edge cases ---
@respx.mock
def test_no_in_flight_reports_returns_an_empty_payload(risk_client):
    route = respx.post(f"{AI_BASE}/ml/predict-delay/batch").mock(
        return_value=httpx.Response(200, json=batch())
    )

    response = risk_client.get("/api/ai/risk-analytics")

    assert response.status_code == 200
    body = response.json()
    assert body["reports_scored"] == 0
    assert body["reports"] == []
    assert body["risk_distribution"] == {"low": 0, "medium": 0, "high": 0}
    assert route.call_count == 0


@respx.mock
def test_untrained_model_returns_503(risk_client, seeded):
    respx.post(f"{AI_BASE}/ml/predict-delay/batch").mock(
        return_value=httpx.Response(
            503, json={"error": {"code": "ML_MODEL_NOT_TRAINED", "message": "train it"}}
        )
    )

    response = risk_client.get("/api/ai/risk-analytics")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ML_MODEL_NOT_TRAINED"


@respx.mock
def test_ai_outage_does_not_affect_reports(risk_client, seeded):
    respx.post(f"{AI_BASE}/ml/predict-delay/batch").mock(
        side_effect=httpx.ConnectError("refused")
    )

    analytics = risk_client.get("/api/ai/risk-analytics")

    assert analytics.status_code == 503
    assert risk_client.get("/api/reports").status_code == 200
    assert risk_client.get("/api/dashboard").status_code == 200


def test_limit_bounds_are_enforced(risk_client):
    assert risk_client.get("/api/ai/risk-analytics", params={"limit": 0}).status_code == 422
    assert (
        risk_client.get("/api/ai/risk-analytics", params={"limit": 501}).status_code
        == 422
    )


@respx.mock
def test_urgent_reports_are_scored_first(risk_client, seeded):
    """The table is a work queue: urgent work must not fall off a low limit."""
    route = respx.post(f"{AI_BASE}/ml/predict-delay/batch").mock(
        return_value=httpx.Response(200, json=batch(prediction(0.5, "medium")))
    )

    risk_client.get("/api/ai/risk-analytics", params={"limit": 1})

    sent = json.loads(route.calls[0].request.content)["items"]
    assert sent[0]["priority"] == "urgent"


def test_in_flight_query_avoids_mysql_incompatible_sql(db):
    """Regression: NULLS LAST compiles on SQLite but is a syntax error on MySQL.

    The unit suite runs on SQLite, so the portability of the ordering has to be
    asserted on the compiled statement rather than discovered in production.
    """
    from sqlalchemy import select
    from sqlalchemy.dialects import mysql

    from app.models import Report
    from app.services.risk_service import IN_FLIGHT_STATUSES

    statement = (
        select(Report)
        .where(Report.status.in_(IN_FLIGHT_STATUSES))
        .order_by(
            (Report.priority == "urgent").desc(),
            Report.result_due_at.is_(None).asc(),
            Report.result_due_at.asc(),
            Report.id.desc(),
        )
    )

    compiled = str(statement.compile(dialect=mysql.dialect()))

    assert "NULLS LAST" not in compiled.upper()
