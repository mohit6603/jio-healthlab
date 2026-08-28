"""Operational risk analytics.

Joins live report data to model predictions. The queue features the model
needs -- how many requests are waiting at a branch, how many of those are
urgent -- are computed from the database rather than supplied by the caller,
so the scores reflect the lab's actual state.

Only in-flight requests are scored. A report already ``ready`` or
``delivered`` cannot become late.

No patient identifier is sent to the AI service or returned here: rows are
keyed by report id, and everything else is operational metadata. This is the
same boundary the explanation endpoint uses.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.logging import get_logger
from ..models import Report

logger = get_logger(__name__)

#: Statuses that can still miss their turnaround target.
IN_FLIGHT_STATUSES = ("registered", "collected", "processing", "review")

#: Probability at or above which a report is counted as predicted late. Matches
#: the AI service's default classification threshold.
LATE_THRESHOLD = 0.5

#: Risk bands considered "at risk" for the headline card.
AT_RISK_LEVELS = frozenset({"medium", "high"})


@dataclass(slots=True)
class BranchLoad:
    """Live queue state for one branch."""

    queue_size: int = 0
    urgent_count: int = 0


@dataclass(slots=True)
class GroupRisk:
    """Aggregated risk for one branch or test type."""

    label: str
    count: int = 0
    total_probability: float = 0.0
    high_risk: int = 0

    @property
    def average_probability(self) -> float:
        return round(self.total_probability / self.count, 4) if self.count else 0.0


@dataclass(slots=True)
class RiskAnalytics:
    """Everything the analytics page renders."""

    generated_at: str
    model_version: str = ""
    synthetic_model: bool = True
    reports_scored: int = 0
    at_risk: int = 0
    high_risk: int = 0
    predicted_late: int = 0
    average_probability: float = 0.0
    highest_risk_branch: dict[str, Any] | None = None
    risk_distribution: dict[str, int] = field(default_factory=dict)
    by_branch: list[dict[str, Any]] = field(default_factory=list)
    by_test_type: list[dict[str, Any]] = field(default_factory=list)
    reports: list[dict[str, Any]] = field(default_factory=list)


def branch_load(db: Session) -> dict[str, BranchLoad]:
    """Count in-flight and urgent in-flight requests per branch.

    Two grouped queries, not one per report.
    """
    loads: dict[str, BranchLoad] = defaultdict(BranchLoad)

    totals = db.execute(
        select(Report.lab_branch, func.count(Report.id))
        .where(Report.status.in_(IN_FLIGHT_STATUSES))
        .group_by(Report.lab_branch)
    ).all()
    for branch, count in totals:
        loads[str(branch or "Unknown")].queue_size = int(count)

    urgent = db.execute(
        select(Report.lab_branch, func.count(Report.id))
        .where(
            Report.status.in_(IN_FLIGHT_STATUSES),
            Report.priority == "urgent",
        )
        .group_by(Report.lab_branch)
    ).all()
    for branch, count in urgent:
        loads[str(branch or "Unknown")].urgent_count = int(count)

    return dict(loads)


def in_flight_reports(db: Session, limit: int) -> list[Report]:
    """Requests that can still miss their target, most urgent first."""
    stmt = (
        select(Report)
        .where(Report.status.in_(IN_FLIGHT_STATUSES))
        .order_by(
            # Urgent first, then soonest due, then newest.
            (Report.priority == "urgent").desc(),
            # NULLS LAST is not valid MySQL. Sorting on the IS NULL predicate
            # first is portable across MySQL, PostgreSQL and SQLite.
            Report.result_due_at.is_(None).asc(),
            Report.result_due_at.asc(),
            Report.id.desc(),
        )
        .limit(limit)
    )
    return list(db.execute(stmt).scalars())


def build_features(report: Report, loads: dict[str, BranchLoad]) -> dict[str, Any]:
    """Build a non-identifying feature payload for one report."""
    branch = report.lab_branch or "Unknown"
    load = loads.get(branch, BranchLoad())
    created = report.created_at or datetime.now(UTC).replace(tzinfo=None)

    payload: dict[str, Any] = {
        "test_type": report.test_type,
        "priority": report.priority or "routine",
        "created_hour": created.hour,
        "created_day_of_week": created.weekday(),
        "queue_size": load.queue_size,
        "urgent_report_count": min(load.urgent_count, load.queue_size),
    }
    if report.lab_branch:
        payload["branch"] = report.lab_branch
    if report.city:
        payload["city"] = report.city
    return payload


def aggregate(
    reports: list[Report],
    predictions: list[dict[str, Any]],
    *,
    model_version: str,
    synthetic_model: bool,
) -> RiskAnalytics:
    """Combine reports and their scores into the analytics payload."""
    analytics = RiskAnalytics(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        model_version=model_version,
        synthetic_model=synthetic_model,
        reports_scored=len(predictions),
    )

    distribution = {"low": 0, "medium": 0, "high": 0}
    by_branch: dict[str, GroupRisk] = {}
    by_test: dict[str, GroupRisk] = {}
    total_probability = 0.0

    for report, prediction in zip(reports, predictions, strict=True):
        probability = float(prediction.get("delay_probability", 0.0))
        level = str(prediction.get("risk_level", "low"))

        total_probability += probability
        distribution[level] = distribution.get(level, 0) + 1
        if level in AT_RISK_LEVELS:
            analytics.at_risk += 1
        if level == "high":
            analytics.high_risk += 1
        if probability >= LATE_THRESHOLD:
            analytics.predicted_late += 1

        for bucket, key in (
            (by_branch, report.lab_branch or "Unknown"),
            (by_test, report.test_type),
        ):
            group = bucket.setdefault(key, GroupRisk(label=key))
            group.count += 1
            group.total_probability += probability
            if level == "high":
                group.high_risk += 1

        analytics.reports.append(
            {
                "report_id": report.id,
                "test_type": report.test_type,
                "branch": report.lab_branch,
                "city": report.city,
                "priority": report.priority,
                "status": report.status,
                "result_due_at": (
                    report.result_due_at.isoformat() if report.result_due_at else None
                ),
                "delay_probability": round(probability, 4),
                "risk_level": level,
            }
        )

    if predictions:
        analytics.average_probability = round(total_probability / len(predictions), 4)

    analytics.risk_distribution = distribution
    analytics.by_branch = _rank(by_branch)
    analytics.by_test_type = _rank(by_test)

    if analytics.by_branch:
        top = analytics.by_branch[0]
        analytics.highest_risk_branch = {
            "branch": top["label"],
            "average_probability": top["average_probability"],
            "reports": top["count"],
            "high_risk": top["high_risk"],
        }

    # Riskiest first: the table is a work queue, not a report listing.
    analytics.reports.sort(key=lambda row: row["delay_probability"], reverse=True)
    return analytics


def _rank(groups: dict[str, GroupRisk]) -> list[dict[str, Any]]:
    """Order groups by average risk, worst first."""
    return [
        {
            "label": group.label,
            "count": group.count,
            "average_probability": group.average_probability,
            "high_risk": group.high_risk,
        }
        for group in sorted(
            groups.values(),
            key=lambda item: (item.average_probability, item.count),
            reverse=True,
        )
    ]


def empty_analytics() -> RiskAnalytics:
    """Nothing in flight: a valid, empty payload rather than an error."""
    return RiskAnalytics(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        risk_distribution={"low": 0, "medium": 0, "high": 0},
    )
