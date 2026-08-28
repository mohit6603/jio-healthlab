"""Training pipeline, evaluation and registry tests."""

import json

import numpy as np
import pytest

from app.ml.dataset import generate_dataset
from app.ml.evaluate import evaluate_classifier, format_metrics
from app.ml.registry import ModelMetadata, ModelRegistry, make_version
from app.ml.train import candidate_models, train, validate_dataset


# ------------------------------------------------------------- evaluate ---
def test_perfect_predictions_score_perfectly():
    metrics = evaluate_classifier([0, 0, 1, 1], [0.01, 0.02, 0.98, 0.99])

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["false_positive"] == 0


def test_random_predictions_score_near_chance():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 2000)
    y_prob = rng.random(2000)

    assert 0.45 < evaluate_classifier(y_true, y_prob)["roc_auc"] < 0.55


def test_confusion_matrix_counts_are_consistent():
    metrics = evaluate_classifier([0, 0, 1, 1], [0.9, 0.1, 0.9, 0.1])
    matrix = metrics["confusion_matrix"]

    assert sum(matrix.values()) == metrics["support"]["total"] == 4
    assert matrix["true_positive"] == 1
    assert matrix["false_positive"] == 1


def test_threshold_shifts_precision_and_recall():
    y_true = [0, 0, 1, 1]
    y_prob = [0.3, 0.6, 0.55, 0.9]

    lenient = evaluate_classifier(y_true, y_prob, threshold=0.4)
    strict = evaluate_classifier(y_true, y_prob, threshold=0.8)

    assert lenient["recall"] >= strict["recall"]


def test_brier_score_rewards_calibration():
    confident_right = evaluate_classifier([0, 1], [0.02, 0.98])["brier_score"]
    confident_wrong = evaluate_classifier([0, 1], [0.98, 0.02])["brier_score"]

    assert confident_right < confident_wrong


def test_format_metrics_renders_every_headline_number():
    rendered = format_metrics(evaluate_classifier([0, 1], [0.1, 0.9]))

    for label in ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion"):
        assert label in rendered


# ------------------------------------------------------------ validation ---
def test_validate_accepts_a_generated_dataset():
    validate_dataset(generate_dataset(300, seed=5))


def test_validate_rejects_missing_columns():
    frame = generate_dataset(100, seed=5).drop(columns=["queue_size"])

    with pytest.raises(ValueError, match="missing columns"):
        validate_dataset(frame)


def test_validate_rejects_nulls():
    frame = generate_dataset(100, seed=5)
    frame.loc[0, "queue_size"] = None

    with pytest.raises(ValueError, match="nulls"):
        validate_dataset(frame)


def test_validate_rejects_a_single_class():
    frame = generate_dataset(100, seed=5)
    frame["delayed"] = 0

    with pytest.raises(ValueError, match="binary"):
        validate_dataset(frame)


# -------------------------------------------------------------- registry ---
def test_version_is_sortable_and_prefixed():
    version = make_version()

    assert version.startswith("v")
    assert len(version) == 15


def test_registry_round_trip(tmp_path):
    registry = ModelRegistry(tmp_path)
    metadata = ModelMetadata(
        model_name="report-delay-classifier",
        version="v20260101000000",
        algorithm="logistic_regression",
        trained_at="2026-01-01T00:00:00+00:00",
        metrics={"roc_auc": 0.86},
    )

    registry.save({"fake": "model"}, metadata)
    model, loaded = registry.load()

    assert model == {"fake": "model"}
    assert loaded.version == "v20260101000000"
    assert loaded.metrics["roc_auc"] == 0.86
    assert loaded.artifact_path.endswith(".joblib")


def test_registry_marks_the_saved_model_current(tmp_path):
    registry = ModelRegistry(tmp_path)
    for version in ("v20260101000000", "v20260202000000"):
        registry.save(
            {"v": version},
            ModelMetadata(
                model_name="m", version=version, algorithm="a", trained_at="t"
            ),
        )

    assert registry.current_version() == "v20260202000000"
    assert json.loads(registry.pointer_path.read_text())["version"] == (
        "v20260202000000"
    )


def test_registry_lists_versions_oldest_first(tmp_path):
    registry = ModelRegistry(tmp_path)
    for version in ("v20260202000000", "v20260101000000"):
        registry.save(
            {},
            ModelMetadata(
                model_name="m", version=version, algorithm="a", trained_at="t"
            ),
        )

    assert registry.list_versions() == ["v20260101000000", "v20260202000000"]


def test_registry_falls_back_when_the_pointer_is_corrupt(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.save(
        {},
        ModelMetadata(
            model_name="m", version="v20260101000000", algorithm="a", trained_at="t"
        ),
    )
    registry.pointer_path.write_text("not json")

    assert registry.current_version() == "v20260101000000"


def test_loading_from_an_empty_registry_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"app\.ml\.train"):
        ModelRegistry(tmp_path).load()


# -------------------------------------------------------------- training ---
def test_candidate_set_covers_the_three_required_algorithms():
    assert set(candidate_models(0)) == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }


@pytest.fixture(name="trained", scope="module")
def trained_fixture():
    """One real training run, reused across assertions -- it is slow."""
    return train(samples=1500, seed=11, dry_run=True, quiet=True)


def test_training_selects_one_of_the_candidates(trained):
    assert trained.algorithm in candidate_models(0)


def test_training_compares_all_candidates(trained):
    assert len(trained.candidates) == 3
    assert all("validation" in item for item in trained.candidates)


def test_training_reports_the_full_metric_set(trained):
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in trained.metrics
    assert "confusion_matrix" in trained.metrics


def test_model_beats_chance(trained):
    """A working pipeline must extract real signal from the generator."""
    assert trained.metrics["roc_auc"] > 0.65


def test_metadata_records_the_feature_schema(trained):
    schema = trained.feature_schema

    assert schema["categorical"]
    assert schema["numeric"]
    assert schema["derived"]
    assert "delayed" not in schema["input_columns"]


def test_metadata_records_provenance(trained):
    assert trained.synthetic_data is True
    assert trained.dataset["seed"] == 11
    assert trained.dataset["dataset_version"] == "synthetic-v1"
    assert "synthetic" in trained.notes.lower()


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "model_dir", str(tmp_path), raising=False)

    train(samples=800, seed=3, dry_run=True, quiet=True)

    assert list(tmp_path.iterdir()) == []


def test_training_persists_a_loadable_model(tmp_path, monkeypatch):
    import app.ml.train as train_module

    monkeypatch.setattr(
        train_module, "get_settings", lambda: _settings_with_dir(tmp_path)
    )

    metadata = train(samples=1200, seed=5, quiet=True)
    model, loaded = ModelRegistry(tmp_path).load()

    assert loaded.version == metadata.version
    assert hasattr(model, "predict_proba")


def _settings_with_dir(directory):
    from app.config import Settings

    return Settings(model_dir=str(directory), log_json=False)
