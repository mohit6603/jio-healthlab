"""Feature engineering tests."""

import pandas as pd
import pytest

from app.ml.dataset import TARGET_COLUMN, generate_dataset
from app.ml.features import (
    CATEGORICAL_FEATURES,
    DERIVED_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    add_derived_features,
    build_pipeline,
    model_input_columns,
)


@pytest.fixture(name="raw", scope="module")
def raw_fixture() -> pd.DataFrame:
    return generate_dataset(400, seed=3)


def test_target_name_matches_the_dataset():
    assert TARGET == TARGET_COLUMN


def test_derivation_is_pure(raw):
    before = raw.copy()

    add_derived_features(raw)

    assert raw.equals(before)


def test_derived_columns_are_added(raw):
    enriched = add_derived_features(raw)

    assert set(DERIVED_FEATURES).issubset(enriched.columns)


def test_headroom_ratio_is_historical_over_expected(raw):
    enriched = add_derived_features(raw)
    row = enriched.iloc[0]

    assert row["tat_headroom_ratio"] == pytest.approx(
        row["historical_avg_tat_minutes"] / row["expected_tat_minutes"], abs=1e-3
    )


def test_urgent_share_is_bounded(raw):
    enriched = add_derived_features(raw)

    assert enriched["urgent_share_of_queue"].between(0, 1).all()


def test_off_hours_flag_matches_the_hour(raw):
    enriched = add_derived_features(raw)
    night = enriched[(enriched["created_hour"] >= 18) | (enriched["created_hour"] < 6)]
    day = enriched[(enriched["created_hour"] >= 6) & (enriched["created_hour"] < 18)]

    assert (night["off_hours_arrival"] == 1).all()
    assert (day["off_hours_arrival"] == 0).all()


def test_zero_queue_does_not_divide_by_zero():
    frame = pd.DataFrame(
        {
            "historical_avg_tat_minutes": [100.0],
            "expected_tat_minutes": [200.0],
            "urgent_report_count": [0],
            "queue_size": [0],
            "created_hour": [10],
        }
    )

    enriched = add_derived_features(frame)

    assert enriched["urgent_share_of_queue"].iloc[0] == 0.0


def test_input_columns_are_the_full_set():
    columns = model_input_columns()

    assert columns == CATEGORICAL_FEATURES + NUMERIC_FEATURES + DERIVED_FEATURES


def test_target_is_never_an_input():
    assert TARGET not in model_input_columns()


def test_pipeline_fits_and_predicts_probabilities(raw):
    from sklearn.linear_model import LogisticRegression

    enriched = add_derived_features(raw)
    pipeline = build_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(enriched[model_input_columns()], enriched[TARGET])

    probabilities = pipeline.predict_proba(enriched[model_input_columns()])[:, 1]

    assert len(probabilities) == len(enriched)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_unseen_category_does_not_crash_inference(raw):
    """A new branch must not break a deployed model."""
    from sklearn.linear_model import LogisticRegression

    enriched = add_derived_features(raw)
    pipeline = build_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(enriched[model_input_columns()], enriched[TARGET])

    unseen = enriched.head(1).copy()
    unseen["branch"] = "Branch That Did Not Exist At Training Time"

    probability = pipeline.predict_proba(unseen[model_input_columns()])[:, 1]

    assert 0.0 <= float(probability[0]) <= 1.0
