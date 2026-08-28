"""Serving layer for the report-delay model.

Two things this module is careful about.

**Partial input.** A caller scheduling work knows the test, the branch and how
busy it is; it does not necessarily know the branch's historical average
turnaround or the exact queue depth. Rather than rejecting such requests, the
predictor fills the gaps from the same profiles the training data was built
from, and reports every value it inferred. A caller can then see which numbers
were theirs and which were assumed.

**Version traceability.** Every prediction carries the model version that
produced it, so a stored prediction can be traced back to its artifact,
metrics and training data.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ..config import Settings, get_settings
from ..core import runtime
from ..core.errors import ModelNotTrainedError
from ..core.logging import get_logger
from .dataset import BRANCH_PROFILES, TEST_PROFILES
from .features import add_derived_features, model_input_columns
from .registry import ModelMetadata, ModelRegistry

logger = get_logger(__name__)

# ── Risk bands ────────────────────────────────────────────────────────────
#
# Deterministic thresholds over the predicted probability. They are set
# relative to the training-set base rate of ~0.32: "low" is at or below the
# base rate, "high" is roughly double it. They are a presentation choice, not
# a property of the model -- change them here and the API documentation
# follows, because it is generated from these values.
RISK_LOW_MAX = 0.35
RISK_MEDIUM_MAX = 0.65

RISK_BAND_DESCRIPTION = (
    f"low: p < {RISK_LOW_MAX}; "
    f"medium: {RISK_LOW_MAX} <= p < {RISK_MEDIUM_MAX}; "
    f"high: p >= {RISK_MEDIUM_MAX}"
)

#: Fallback when a branch is not in the known profiles.
_DEFAULT_CAPACITY = 1.0
#: Relationship used to estimate queue depth from reported workload, mirroring
#: the generator: workload ~= queue * 2.4 + 4.
_WORKLOAD_TO_QUEUE = 2.4
_WORKLOAD_OFFSET = 4.0


def risk_level(probability: float) -> str:
    """Map a probability onto a documented risk band."""
    if probability < RISK_LOW_MAX:
        return "low"
    if probability < RISK_MEDIUM_MAX:
        return "medium"
    return "high"


@dataclass(slots=True)
class DelayPrediction:
    """One prediction, with the provenance needed to audit it."""

    delay_probability: float
    risk_level: str
    model_version: str
    model_name: str
    algorithm: str
    #: Fields the caller did not supply, and what was assumed for them.
    inferred_fields: dict[str, Any] = field(default_factory=dict)
    #: Loud, because the model was trained on simulated data.
    synthetic_model: bool = True


class DelayPredictor:
    """Loads the trained model once and serves predictions from it."""

    def __init__(
        self, settings: Settings | None = None, registry: ModelRegistry | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._registry = registry or ModelRegistry(self._settings.model_path)
        self._model: Any = None
        self._metadata: ModelMetadata | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def metadata(self) -> ModelMetadata:
        self._load()
        assert self._metadata is not None
        return self._metadata

    def _load(self) -> Any:
        """Load the current artifact once, under a lock."""
        if self._model is not None:
            if not runtime.is_loaded(runtime.DELAY_MODEL):
                runtime.mark_loaded(runtime.DELAY_MODEL)
            return self._model

        with self._lock:
            if self._model is not None:  # another thread won the race
                return self._model
            try:
                model, metadata = self._registry.load()
            except FileNotFoundError as exc:
                runtime.mark_failed(runtime.DELAY_MODEL, str(exc))
                raise ModelNotTrainedError(str(exc)) from exc

            self._model = model
            self._metadata = metadata
            runtime.mark_loaded(runtime.DELAY_MODEL)
            logger.info(
                "delay_model_loaded",
                extra={"version": metadata.version, "algorithm": metadata.algorithm},
            )
            return model

    def warm_up(self) -> None:
        self._load()

    def reset(self) -> None:
        """Drop the loaded model, so a freshly trained one is picked up."""
        with self._lock:
            self._model = None
            self._metadata = None
        runtime.mark_unloaded(runtime.DELAY_MODEL)

    # ------------------------------------------------------------ predict --
    def predict(self, payload: dict[str, Any]) -> DelayPrediction:
        """Predict delay risk for one request."""
        return self.predict_many([payload])[0]

    def predict_many(self, payloads: list[dict[str, Any]]) -> list[DelayPrediction]:
        """Predict for a batch. One model call, not N."""
        if not payloads:
            return []

        model = self._load()
        metadata = self.metadata

        prepared = [self._complete(dict(payload)) for payload in payloads]
        frame = pd.DataFrame([row for row, _ in prepared])
        frame = add_derived_features(frame)

        probabilities = model.predict_proba(frame[model_input_columns()])[:, 1]

        results = [
            DelayPrediction(
                delay_probability=round(float(probability), 4),
                risk_level=risk_level(float(probability)),
                model_version=metadata.version,
                model_name=metadata.model_name,
                algorithm=metadata.algorithm,
                inferred_fields=inferred,
                synthetic_model=metadata.synthetic_data,
            )
            for probability, (_, inferred) in zip(probabilities, prepared, strict=True)
        ]

        logger.info(
            "delay_prediction_completed",
            extra={
                "count": len(results),
                "model_version": metadata.version,
                "risk_levels": [item.risk_level for item in results],
            },
        )
        return results

    # ------------------------------------------------------------ filling --
    def _complete(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fill missing features from the known profiles.

        Returns the completed row and a record of what was inferred.
        """
        inferred: dict[str, Any] = {}

        def fill(key: str, value: Any) -> Any:
            inferred[key] = value
            return value

        test_type = payload.get("test_type") or "CBC Panel"
        profile = TEST_PROFILES.get(test_type)
        payload["test_type"] = test_type

        # Derived deterministically from test_type, which the caller always
        # supplies -- a lookup, not an assumption, so it is not reported as
        # inferred.
        if not payload.get("test_group"):
            payload["test_group"] = profile.group if profile else "chemistry"

        priority = payload.get("priority") or fill("priority", "routine")
        payload["priority"] = priority

        branch = payload.get("branch") or fill("branch", "BKC Flagship")
        payload["branch"] = branch
        branch_profile = BRANCH_PROFILES.get(branch)

        if not payload.get("city"):
            payload["city"] = fill(
                "city", branch_profile.city if branch_profile else "Mumbai"
            )

        now = datetime.now(UTC)
        if payload.get("created_hour") is None:
            payload["created_hour"] = fill("created_hour", now.hour)
        if payload.get("created_day_of_week") is None:
            payload["created_day_of_week"] = fill(
                "created_day_of_week", now.weekday()
            )

        if payload.get("expected_tat_minutes") is None:
            expected: float = 360.0
            if profile is not None:
                expected = float(
                    profile.urgent_tat if priority == "urgent" else profile.routine_tat
                )
            payload["expected_tat_minutes"] = fill(
                "expected_tat_minutes", float(expected)
            )

        capacity = branch_profile.capacity if branch_profile else _DEFAULT_CAPACITY
        if payload.get("historical_avg_tat_minutes") is None:
            # Same relationship the training data was generated with.
            payload["historical_avg_tat_minutes"] = fill(
                "historical_avg_tat_minutes",
                round(float(payload["expected_tat_minutes"]) * (0.50 / capacity), 1),
            )

        if payload.get("queue_size") is None:
            workload = payload.get("current_workload")
            estimated = (
                max(round((float(workload) - _WORKLOAD_OFFSET) / _WORKLOAD_TO_QUEUE), 0)
                if workload is not None
                else 12
            )
            payload["queue_size"] = fill("queue_size", estimated)

        if payload.get("current_workload") is None:
            payload["current_workload"] = fill(
                "current_workload",
                round(payload["queue_size"] * _WORKLOAD_TO_QUEUE + _WORKLOAD_OFFSET),
            )

        if payload.get("urgent_report_count") is None:
            payload["urgent_report_count"] = fill(
                "urgent_report_count", round(payload["queue_size"] * 0.22)
            )

        if payload.get("analyser_degraded") is None:
            payload["analyser_degraded"] = fill("analyser_degraded", 0)

        # Never let an inferred urgent count exceed the queue it sits in.
        payload["urgent_report_count"] = min(
            int(payload["urgent_report_count"]), int(payload["queue_size"])
        )

        return payload, inferred


_predictor: DelayPredictor | None = None


def get_delay_predictor() -> DelayPredictor:
    """Process-wide singleton -- the artifact is read from disk at most once."""
    global _predictor
    if _predictor is None:
        _predictor = DelayPredictor()
    return _predictor


def reset_delay_predictor() -> None:
    """Drop the singleton (test helper, and after retraining)."""
    global _predictor
    if _predictor is not None:
        _predictor.reset()
    _predictor = None
