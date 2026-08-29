"""Schemas for the delay-prediction API."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..ml.predictor import RISK_BAND_DESCRIPTION

RiskLevel = Literal["low", "medium", "high"]


class DelayPredictionRequest(BaseModel):
    """State of a laboratory request at intake.

    Only `test_type` is required. Anything omitted is inferred from the branch
    and test profiles the model was trained on, and every inferred value is
    reported back in `inferred_fields`.
    """

    test_type: str = Field(
        description="Test being performed, e.g. `CBC Panel`.", examples=["CBC Panel"]
    )
    priority: Literal["routine", "urgent"] | None = Field(
        default=None, description="Defaults to `routine`."
    )
    branch: str | None = Field(default=None, examples=["Andheri Hub"])
    city: str | None = Field(default=None, description="Defaults to the branch's city.")
    created_hour: int | None = Field(
        default=None, ge=0, le=23, description="Hour of intake. Defaults to now."
    )
    created_day_of_week: int | None = Field(
        default=None, ge=0, le=6, description="Monday is 0. Defaults to today."
    )
    expected_tat_minutes: float | None = Field(
        default=None,
        gt=0,
        description="Target turnaround. Defaults to the documented target for "
        "this test and priority.",
    )
    historical_avg_tat_minutes: float | None = Field(
        default=None, gt=0, description="Recent average for this branch and test."
    )
    queue_size: int | None = Field(
        default=None, ge=0, description="Requests waiting at the workstation."
    )
    current_workload: int | None = Field(
        default=None, ge=0, description="Total open requests at the branch."
    )
    urgent_report_count: int | None = Field(
        default=None, ge=0, description="Urgent requests queued ahead of this one."
    )
    analyser_degraded: int | None = Field(
        default=None, ge=0, le=1, description="1 when an analyser is down or degraded."
    )


class DelayPredictionResponse(BaseModel):
    """Predicted risk that a request misses its turnaround target."""

    delay_probability: float = Field(
        ge=0.0, le=1.0, description="Probability the request misses its target."
    )
    risk_level: RiskLevel = Field(description=RISK_BAND_DESCRIPTION)
    model_version: str = Field(description="Version of the artifact that predicted.")
    model_name: str
    algorithm: str
    inferred_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Values not supplied by the caller, and what was assumed.",
    )
    synthetic_model: bool = Field(
        description=(
            "True when the model was trained on synthetic data. It is a "
            "demonstration of the pipeline, not a validated operational tool."
        )
    )


class BatchDelayPredictionRequest(BaseModel):
    """Score many requests in one call."""

    items: list[DelayPredictionRequest] = Field(min_length=1, max_length=500)


class BatchDelayPredictionResponse(BaseModel):
    predictions: list[DelayPredictionResponse]
    model_version: str
    count: int


class ModelCandidate(BaseModel):
    """A candidate algorithm considered during training."""

    algorithm: str
    validation: dict[str, Any] = Field(default_factory=dict)


class ModelInfoResponse(BaseModel):
    """Metadata for the currently served model."""

    model_name: str
    version: str
    algorithm: str
    trained_at: str
    synthetic_data: bool
    dataset: dict[str, Any] = Field(default_factory=dict)
    feature_schema: dict[str, list[str]] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    candidates: list[ModelCandidate] = Field(default_factory=list)
    selection_metric: str = "roc_auc"
    risk_bands: str = RISK_BAND_DESCRIPTION
    notes: str = ""
