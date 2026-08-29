"""Aggregation queries powering the operations dashboard."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import InstrumentedAttribute, Session

from ..models import Report
from ..schemas.dashboard import DashboardSummary

#: Reports due within this window appear in the "due soon" panel.
DUE_SOON_WINDOW = timedelta(hours=24)

#: Statuses that no longer need lab attention.
COMPLETED_STATUSES = ("delivered", "ready")

_PANEL_LIMIT = 6


def build_summary(db: Session) -> DashboardSummary:
    """Compute the dashboard snapshot in a handful of aggregate queries."""
    total_reports = db.scalar(select(func.count(Report.id))) or 0
    ready_reports = (
        db.scalar(select(func.count(Report.id)).where(Report.status == "ready")) or 0
    )
    urgent_reports = (
        db.scalar(select(func.count(Report.id)).where(Report.priority == "urgent")) or 0
    )
    avg_age = db.scalar(select(func.avg(Report.age)))
    unique_tests = db.scalar(select(func.count(func.distinct(Report.test_type)))) or 0
    latest_report_id = db.scalar(select(func.max(Report.id)))

    due_cutoff = datetime.now(UTC).replace(tzinfo=None) + DUE_SOON_WINDOW
    due_soon = list(
        db.execute(
            select(Report)
            .where(
                Report.result_due_at.is_not(None),
                Report.result_due_at <= due_cutoff,
                Report.status.notin_(COMPLETED_STATUSES),
            )
            .order_by(Report.result_due_at.asc())
            .limit(_PANEL_LIMIT)
        ).scalars()
    )

    recent_reports = list(
        db.execute(
            select(Report).order_by(Report.created_at.desc()).limit(_PANEL_LIMIT)
        ).scalars()
    )

    return DashboardSummary(
        total_reports=total_reports,
        ready_reports=ready_reports,
        urgent_reports=urgent_reports,
        avg_age=round(float(avg_age), 1) if avg_age is not None else None,
        unique_tests=unique_tests,
        latest_report_id=latest_report_id,
        due_soon=due_soon,
        recent_reports=recent_reports,
        by_status=count_by(db, Report.status),
        by_test_type=count_by(db, Report.test_type),
        by_city=count_by(db, Report.city),
    )


def count_by(db: Session, column: InstrumentedAttribute) -> dict[str, int]:
    """Group reports by ``column`` and return ``{label: count}``."""
    rows = db.execute(select(column, func.count(Report.id)).group_by(column)).all()
    return {str(label or "Unknown"): int(count) for label, count in rows}
