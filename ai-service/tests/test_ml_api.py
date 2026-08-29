"""POST /ml/predict-delay and related endpoint tests."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.ml.predictor import DelayPredictor, get_delay_predictor
from app.ml.registry import ModelRegistry
from app.ml.train import train


@pytest.fixture(name="model_dir", scope="module")
def model_dir_fixture(tmp_path_factory):
    directory = tmp_path_factory.mktemp("api-model")
    import app.ml.train as train_module

    settings = Settings(model_dir=str(directory), log_json=False)
    original = train_module.get_settings
    train_module.get_settings = lambda: settings
    try:
        train(samples=2000, seed=17, quiet=True)
    finally:
        train_module.get_settings = original
    return directory


@pytest.fixture(name="ml_client")
def ml_client_fixture(client, model_dir):
    predictor = DelayPredictor(
        settings=Settings(model_dir=str(model_dir), log_json=False),
        registry=ModelRegistry(model_dir),
    )
    client.app.dependency_overrides[get_delay_predictor] = lambda: predictor
    return client


# ------------------------------------------------------------------ happy ---
def test_predict_returns_probability_risk_and_version(ml_client):
    response = ml_client.post("/ml/predict-delay", json={"test_type": "CBC Panel"})

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["delay_probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["model_version"].startswith("v")


def test_response_matches_the_documented_contract(ml_client):
    body = ml_client.post("/ml/predict-delay", json={"test_type": "CBC Panel"}).json()

    assert {"delay_probability", "risk_level", "model_version"} <= set(body)


def test_synthetic_provenance_is_surfaced(ml_client):
    body = ml_client.post("/ml/predict-delay", json={"test_type": "CBC Panel"}).json()

    assert body["synthetic_model"] is True


def test_full_payload_is_accepted(ml_client):
    response = ml_client.post(
        "/ml/predict-delay",
        json={
            "test_type": "CBC Panel",
            "priority": "urgent",
            "branch": "Andheri Hub",
            "city": "Mumbai",
            "created_hour": 19,
            "created_day_of_week": 0,
            "expected_tat_minutes": 120,
            "historical_avg_tat_minutes": 95,
            "queue_size": 30,
            "current_workload": 45,
            "urgent_report_count": 8,
            "analyser_degraded": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["inferred_fields"] == {}


def test_inferred_fields_are_returned(ml_client):
    body = ml_client.post("/ml/predict-delay", json={"test_type": "CBC Panel"}).json()

    assert "branch" in body["inferred_fields"]
    assert "queue_size" in body["inferred_fields"]


def test_pressure_raises_the_probability(ml_client):
    quiet = ml_client.post(
        "/ml/predict-delay",
        json={"test_type": "CBC Panel", "branch": "BKC Flagship", "queue_size": 2},
    ).json()
    busy = ml_client.post(
        "/ml/predict-delay",
        json={
            "test_type": "CBC Panel",
            "branch": "Salt Lake",
            "queue_size": 40,
            "urgent_report_count": 14,
            "analyser_degraded": 1,
        },
    ).json()

    assert busy["delay_probability"] > quiet["delay_probability"]


# ------------------------------------------------------------- validation ---
def test_test_type_is_required(ml_client):
    response = ml_client.post("/ml/predict-delay", json={})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize(
    "payload",
    [
        {"test_type": "CBC Panel", "created_hour": 24},
        {"test_type": "CBC Panel", "created_day_of_week": 7},
        {"test_type": "CBC Panel", "queue_size": -1},
        {"test_type": "CBC Panel", "expected_tat_minutes": 0},
        {"test_type": "CBC Panel", "analyser_degraded": 2},
        {"test_type": "CBC Panel", "priority": "immediately"},
    ],
)
def test_out_of_range_values_are_rejected(ml_client, payload):
    assert ml_client.post("/ml/predict-delay", json=payload).status_code == 422


# ------------------------------------------------------------------ batch ---
def test_batch_scores_every_item(ml_client):
    response = ml_client.post(
        "/ml/predict-delay/batch",
        json={
            "items": [
                {"test_type": "CBC Panel", "queue_size": 4},
                {"test_type": "Thyroid", "queue_size": 30},
                {"test_type": "MRI Scan", "priority": "urgent"},
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3
    assert body["model_version"].startswith("v")


def test_batch_agrees_with_single_prediction(ml_client):
    payload = {"test_type": "Thyroid", "queue_size": 25, "created_hour": 9}

    single = ml_client.post("/ml/predict-delay", json=payload).json()
    batched = ml_client.post(
        "/ml/predict-delay/batch", json={"items": [payload]}
    ).json()

    assert batched["predictions"][0]["delay_probability"] == pytest.approx(
        single["delay_probability"]
    )


def test_empty_batch_is_rejected(ml_client):
    assert (
        ml_client.post("/ml/predict-delay/batch", json={"items": []}).status_code == 422
    )


def test_oversized_batch_is_rejected(ml_client):
    items = [{"test_type": "CBC Panel"}] * 501

    assert (
        ml_client.post(
            "/ml/predict-delay/batch", json={"items": items}
        ).status_code
        == 422
    )


# ------------------------------------------------------------- model info ---
def test_model_info_reports_metrics_and_candidates(ml_client):
    body = ml_client.get("/ml/model").json()

    assert body["version"].startswith("v")
    assert body["synthetic_data"] is True
    assert body["metrics"]["roc_auc"] > 0.6
    assert len(body["candidates"]) == 3
    assert body["selection_metric"] == "roc_auc"


def test_model_info_documents_the_risk_bands(ml_client):
    body = ml_client.get("/ml/model").json()

    assert "low" in body["risk_bands"]
    assert "high" in body["risk_bands"]


def test_model_info_exposes_the_feature_schema(ml_client):
    schema = ml_client.get("/ml/model").json()["feature_schema"]

    assert "delayed" not in schema["input_columns"]
    assert schema["derived"]


# -------------------------------------------------------------- untrained ---
def test_untrained_model_returns_503(client, tmp_path):
    predictor = DelayPredictor(
        settings=Settings(model_dir=str(tmp_path), log_json=False),
        registry=ModelRegistry(tmp_path),
    )
    client.app.dependency_overrides[get_delay_predictor] = lambda: predictor

    response = client.post("/ml/predict-delay", json={"test_type": "CBC Panel"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ML_MODEL_NOT_TRAINED"
    assert "app.ml.train" in response.json()["error"]["message"]


def test_rag_search_still_works_without_a_trained_model(client, settings, tmp_path):
    """ML and RAG are independent; neither outage should affect the other."""
    from app.rag.embeddings import get_embedder
    from app.rag.vector_store import VectorStore, get_vector_store
    from tests.test_ingestion import StubEmbedder
    from tests.test_vector_store import StubClient

    predictor = DelayPredictor(
        settings=Settings(model_dir=str(tmp_path), log_json=False),
        registry=ModelRegistry(tmp_path),
    )
    client.app.dependency_overrides[get_delay_predictor] = lambda: predictor
    client.app.dependency_overrides[get_vector_store] = lambda: VectorStore(
        settings=settings, client=StubClient(existing=True)
    )
    client.app.dependency_overrides[get_embedder] = lambda: StubEmbedder()

    assert client.post("/rag/search", json={"query": "cbc"}).status_code == 200
