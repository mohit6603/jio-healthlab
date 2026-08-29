"""Feature schema and preprocessing for the delay model.

The feature list is the contract between training and inference. It is stored
with every model artifact, and the predictor validates against it, so a model
can never be served with a different set of columns from the one it learned.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#: Categorical inputs, one-hot encoded.
CATEGORICAL_FEATURES: list[str] = [
    "test_type",
    "test_group",
    "priority",
    "branch",
    "city",
]

#: Numeric inputs, standardised (needed by logistic regression; harmless for
#: the tree ensembles, and keeping one pipeline avoids two code paths).
NUMERIC_FEATURES: list[str] = [
    "created_hour",
    "created_day_of_week",
    "expected_tat_minutes",
    "historical_avg_tat_minutes",
    "queue_size",
    "current_workload",
    "urgent_report_count",
    "analyser_degraded",
]

#: Every column the model consumes, in a stable order.
FEATURE_COLUMNS: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

#: Derived at training time from the columns above.
DERIVED_FEATURES: list[str] = [
    # How much of the target the historical average already consumes. A value
    # near 1.0 means there is no slack before the request is late.
    "tat_headroom_ratio",
    # Queue depth relative to the urgent work sitting ahead of it.
    "urgent_share_of_queue",
    # Requests arriving into the evening or night wait for the next shift.
    "off_hours_arrival",
]

TARGET = "delayed"


def add_derived_features(frame):
    """Add engineered columns. Pure: returns a copy.

    Every derived value is computable from information available at intake, so
    nothing here can leak the outcome.
    """
    enriched = frame.copy()
    enriched["tat_headroom_ratio"] = (
        enriched["historical_avg_tat_minutes"]
        / enriched["expected_tat_minutes"].clip(lower=1.0)
    ).round(4)
    enriched["urgent_share_of_queue"] = (
        enriched["urgent_report_count"] / enriched["queue_size"].clip(lower=1)
    ).round(4)
    enriched["off_hours_arrival"] = (
        (enriched["created_hour"] >= 18) | (enriched["created_hour"] < 6)
    ).astype(int)
    return enriched


def model_input_columns() -> list[str]:
    """Columns the fitted pipeline expects, after derivation."""
    return CATEGORICAL_FEATURES + NUMERIC_FEATURES + DERIVED_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """One-hot the categoricals, standardise the numerics."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES + DERIVED_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_pipeline(estimator) -> Pipeline:
    """Wrap an estimator with the shared preprocessing."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
