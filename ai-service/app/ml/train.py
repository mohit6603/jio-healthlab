"""Training pipeline for the report-delay model.

    generate -> validate -> engineer -> split -> train -> compare
             -> select -> evaluate -> persist

Run it::

    python -m app.ml.train
    python -m app.ml.train --samples 30000 --seed 7
    python -m app.ml.train --dry-run          # report, write nothing

Three splits, not two. Candidate models are compared on a *validation* split;
the metrics recorded and reported come from a *test* split that played no part
in selection. Choosing on the same data you report from inflates the numbers,
and these numbers are already only a statement about a simulation.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ..config import get_settings
from ..core.logging import configure_logging, get_logger
from .dataset import DATASET_VERSION, TARGET_COLUMN, describe_dataset, generate_dataset
from .evaluate import evaluate_classifier, format_metrics
from .features import (
    CATEGORICAL_FEATURES,
    DERIVED_FEATURES,
    NUMERIC_FEATURES,
    add_derived_features,
    build_pipeline,
    model_input_columns,
)
from .registry import ModelMetadata, ModelRegistry, make_version

logger = get_logger(__name__)

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20


def candidate_models(seed: int) -> dict[str, Any]:
    """The estimators to compare.

    XGBoost is deliberately not included: it adds a compiled dependency to the
    image, and on a dataset this size gradient boosting from scikit-learn
    occupies the same niche.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=3,
            subsample=0.9,
            random_state=seed,
        ),
    }


def validate_dataset(frame: pd.DataFrame) -> None:
    """Fail loudly on a raw dataset that cannot train a useful model.

    Runs before feature derivation, so it checks the source columns only.
    """
    required = {*CATEGORICAL_FEATURES, *NUMERIC_FEATURES, TARGET_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    null_counts = frame[sorted(required)].isna().sum()
    if null_counts.any():
        offenders = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Dataset contains nulls: {offenders}")

    classes = sorted(frame[TARGET_COLUMN].unique())
    if classes != [0, 1]:
        raise ValueError(f"Target must be binary 0/1, found {classes}")

    positive_rate = float(frame[TARGET_COLUMN].mean())
    if not 0.02 < positive_rate < 0.98:
        raise ValueError(
            f"Positive rate {positive_rate:.3f} is too extreme to learn from."
        )


def train(
    *,
    samples: int = 12_000,
    seed: int = 42,
    dry_run: bool = False,
    quiet: bool = False,
) -> ModelMetadata:
    """Run the full pipeline and return the selected model's metadata."""
    settings = get_settings()

    def say(message: str = "") -> None:
        if not quiet:
            print(message)

    say(f"Generating {samples} synthetic requests (seed={seed}) …")
    frame = generate_dataset(samples, seed=seed)
    validate_dataset(frame)
    frame = add_derived_features(frame)
    derived_missing = set(DERIVED_FEATURES) - set(frame.columns)
    if derived_missing:  # pragma: no cover - guards a coding error
        raise ValueError(f"Feature derivation failed: {sorted(derived_missing)}")

    summary = describe_dataset(frame)
    say(
        f"  rows={summary['rows']}  positive_rate={summary['positive_rate']}  "
        f"dataset={DATASET_VERSION}  (SYNTHETIC)"
    )

    features = frame[model_input_columns()]
    target = frame[TARGET_COLUMN]

    # Stratified so every split keeps the same class balance.
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, stratify=target, random_state=seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        stratify=y_train_full,
        random_state=seed,
    )
    say(f"  split: train={len(x_train)} val={len(x_val)} test={len(x_test)}\n")

    say("Training candidates …")
    fitted: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []

    for name, estimator in candidate_models(seed).items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        val_metrics = evaluate_classifier(
            y_val, pipeline.predict_proba(x_val)[:, 1]
        )
        fitted[name] = pipeline
        candidates.append({"algorithm": name, "validation": val_metrics})
        say(
            f"  {name:22} val roc_auc={val_metrics['roc_auc']:.4f} "
            f"f1={val_metrics['f1']:.4f}"
        )

    best = max(candidates, key=lambda item: item["validation"]["roc_auc"])
    best_name = str(best["algorithm"])
    say(f"\nSelected: {best_name} (highest validation roc_auc)")

    # Refit the winner on train + validation, then report on the untouched test
    # split. Selection never saw this data.
    final_model = build_pipeline(candidate_models(seed)[best_name])
    final_model.fit(x_train_full, y_train_full)
    test_metrics = evaluate_classifier(
        y_test, final_model.predict_proba(x_test)[:, 1]
    )

    say("\nHeld-out test metrics (selection never saw this split):")
    say(format_metrics(test_metrics))

    metadata = ModelMetadata(
        model_name=settings.delay_model_name,
        version=make_version(),
        algorithm=best_name,
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        dataset={**summary, "seed": seed, "samples": samples},
        feature_schema={
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
            "derived": DERIVED_FEATURES,
            "input_columns": model_input_columns(),
        },
        metrics=test_metrics,
        candidates=candidates,
        artifact_path="",
        synthetic_data=True,
        notes=(
            "Trained on synthetic data generated by app.ml.dataset. Metrics "
            "describe how well the model recovers that generator and are not "
            "evidence of real-world operational accuracy."
        ),
    )

    if dry_run:
        say("\nDry run: nothing written.")
        return metadata

    registry = ModelRegistry(settings.model_path)
    artifact = registry.save(final_model, metadata)
    say(f"\nSaved {metadata.version} -> {artifact}")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.ml.train",
        description="Train the report-delay classifier on synthetic data.",
    )
    parser.add_argument("--samples", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run", action="store_true", help="Train and report, but write nothing."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level=get_settings().log_level, json_output=False)

    try:
        train(samples=args.samples, seed=args.seed, dry_run=args.dry_run)
    except (ValueError, OSError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
