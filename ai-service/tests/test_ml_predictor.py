"""Delay predictor tests.

A small model is trained into a temporary registry for the module, so these
tests never depend on an artifact that happens to exist on the developer's
disk -- and they fail honestly in CI if training breaks.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.core import runtime
from app.core.errors import ModelNotTrainedError
from app.ml.predictor import (
    RISK_LOW_MAX,
    RISK_MEDIUM_MAX,
    DelayPredictor,
    risk_level,
)
from app.ml.registry import ModelRegistry
from app.ml.train import train


@pytest.fixture(name="trained_registry", scope="module")
def trained_registry_fixture(tmp_path_factory):
    """Train once for the module; training is the slow part."""
    directory = tmp_path_factory.mktemp("model-registry")
    import app.ml.train as train_module

    settings = Settings(model_dir=str(directory), log_json=False)
    original = train_module.get_settings
    train_module.get_settings = lambda: settings
    try:
        train(samples=2000, seed=13, quiet=True)
    finally:
        train_module.get_settings = original
    return directory


@pytest.fixture(name="predictor")
def predictor_fixture(trained_registry):
    settings = Settings(model_dir=str(trained_registry), log_json=False)
    return DelayPredictor(settings=settings, registry=ModelRegistry(trained_registry))


# ------------------------------------------------------------ risk bands ---
@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "low"),
        (0.34, "low"),
        (RISK_LOW_MAX, "medium"),
        (0.5, "medium"),
        (0.649, "medium"),
        (RISK_MEDIUM_MAX, "high"),
        (1.0, "high"),
    ],
)
def test_risk_bands_are_deterministic(probability, expected):
    assert risk_level(probability) == expected


def test_bands_are_ordered():
    assert RISK_LOW_MAX < RISK_MEDIUM_MAX


# ------------------------------------------------------------- lifecycle ---
def test_model_is_not_loaded_until_used(predictor):
    assert predictor.is_loaded is False


def test_prediction_loads_the_model(predictor):
    predictor.predict({"test_type": "CBC Panel"})

    assert predictor.is_loaded is True
    assert runtime.is_loaded(runtime.DELAY_MODEL) is True


def test_model_loads_only_once(predictor):
    predictor.predict({"test_type": "CBC Panel"})
    first = predictor._model
    predictor.predict({"test_type": "Thyroid"})

    assert predictor._model is first


def test_missing_artifact_raises_a_typed_error(tmp_path):
    predictor = DelayPredictor(
        settings=Settings(model_dir=str(tmp_path), log_json=False),
        registry=ModelRegistry(tmp_path),
    )

    with pytest.raises(ModelNotTrainedError) as excinfo:
        predictor.predict({"test_type": "CBC Panel"})

    assert excinfo.value.code == "ML_MODEL_NOT_TRAINED"
    assert excinfo.value.status_code == 503


# ------------------------------------------------------------ prediction ---
def test_prediction_shape(predictor):
    result = predictor.predict({"test_type": "CBC Panel"})

    assert 0.0 <= result.delay_probability <= 1.0
    assert result.risk_level in {"low", "medium", "high"}
    assert result.model_version.startswith("v")
    assert result.algorithm
    assert result.synthetic_model is True


def test_risk_level_matches_the_probability(predictor):
    result = predictor.predict({"test_type": "CBC Panel", "queue_size": 30})

    assert result.risk_level == risk_level(result.delay_probability)


def test_prediction_is_deterministic(predictor):
    payload = {"test_type": "Thyroid", "queue_size": 20, "created_hour": 9}

    first = predictor.predict(dict(payload))
    second = predictor.predict(dict(payload))

    assert first.delay_probability == second.delay_probability


# ---------------------------------------------------------- directionality --
def test_a_deep_queue_raises_risk(predictor):
    base = {"test_type": "CBC Panel", "branch": "Salt Lake", "created_hour": 10}

    quiet = predictor.predict({**base, "queue_size": 3, "urgent_report_count": 0})
    busy = predictor.predict({**base, "queue_size": 40, "urgent_report_count": 12})

    assert busy.delay_probability > quiet.delay_probability


def test_a_degraded_analyser_raises_risk(predictor):
    base = {"test_type": "Thyroid", "branch": "Salt Lake", "queue_size": 20}

    healthy = predictor.predict({**base, "analyser_degraded": 0})
    degraded = predictor.predict({**base, "analyser_degraded": 1})

    assert degraded.delay_probability > healthy.delay_probability


def test_a_better_resourced_branch_lowers_risk(predictor):
    base = {"test_type": "CBC Panel", "queue_size": 25, "created_hour": 10}

    weak = predictor.predict({**base, "branch": "Salt Lake"})
    strong = predictor.predict({**base, "branch": "BKC Flagship"})

    assert strong.delay_probability < weak.delay_probability


# --------------------------------------------------------------- filling ---
def test_only_test_type_is_required(predictor):
    result = predictor.predict({"test_type": "CBC Panel"})

    assert result.delay_probability >= 0.0


def test_inferred_fields_are_reported(predictor):
    result = predictor.predict({"test_type": "CBC Panel"})

    assert "priority" in result.inferred_fields
    assert "branch" in result.inferred_fields
    assert "queue_size" in result.inferred_fields


def test_supplied_fields_are_not_reported_as_inferred(predictor):
    result = predictor.predict(
        {
            "test_type": "CBC Panel",
            "priority": "urgent",
            "branch": "Andheri Hub",
            "queue_size": 12,
        }
    )

    assert "priority" not in result.inferred_fields
    assert "branch" not in result.inferred_fields
    assert "queue_size" not in result.inferred_fields


def test_expected_tat_defaults_to_the_documented_target(predictor):
    routine = predictor.predict({"test_type": "CBC Panel", "priority": "routine"})
    urgent = predictor.predict({"test_type": "CBC Panel", "priority": "urgent"})

    assert routine.inferred_fields["expected_tat_minutes"] == 240.0
    assert urgent.inferred_fields["expected_tat_minutes"] == 60.0


def test_city_defaults_from_the_branch(predictor):
    result = predictor.predict({"test_type": "CBC Panel", "branch": "Koregaon Park"})

    assert result.inferred_fields["city"] == "Pune"


def test_queue_is_estimated_from_workload(predictor):
    result = predictor.predict({"test_type": "CBC Panel", "current_workload": 52})

    assert result.inferred_fields["queue_size"] == 20


def test_urgent_count_cannot_exceed_the_queue(predictor):
    result = predictor.predict(
        {"test_type": "CBC Panel", "queue_size": 3, "urgent_report_count": 99}
    )

    assert result.delay_probability >= 0.0


def test_unknown_test_type_does_not_crash(predictor):
    result = predictor.predict({"test_type": "Some Test We Do Not Offer"})

    assert 0.0 <= result.delay_probability <= 1.0


def test_unknown_branch_does_not_crash(predictor):
    result = predictor.predict(
        {"test_type": "CBC Panel", "branch": "Branch Opened Yesterday"}
    )

    assert 0.0 <= result.delay_probability <= 1.0


# ----------------------------------------------------------------- batch ---
def test_batch_matches_individual_predictions(predictor):
    payloads = [
        {"test_type": "CBC Panel", "queue_size": 5},
        {"test_type": "Thyroid", "queue_size": 30},
        {"test_type": "MRI Scan", "priority": "urgent"},
    ]

    batch = predictor.predict_many([dict(item) for item in payloads])
    singles = [predictor.predict(dict(item)) for item in payloads]

    assert [item.delay_probability for item in batch] == [
        item.delay_probability for item in singles
    ]


def test_empty_batch_returns_nothing(predictor):
    assert predictor.predict_many([]) == []


# -------------------------------------------------------------- metadata ---
def test_metadata_is_exposed(predictor):
    metadata = predictor.metadata

    assert metadata.algorithm
    assert metadata.synthetic_data is True
    assert metadata.metrics["roc_auc"] > 0.6
    assert metadata.feature_schema["input_columns"]


def test_reset_forces_a_reload(predictor):
    predictor.predict({"test_type": "CBC Panel"})
    predictor.reset()

    assert predictor.is_loaded is False
    assert runtime.is_loaded(runtime.DELAY_MODEL) is False
