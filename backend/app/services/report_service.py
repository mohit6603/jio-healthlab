"""Business logic for laboratory report CRUD and querying.

Routers stay thin: they validate input and translate results. Every database
access for reports lives here so it can be unit-tested and reused (the AI
report-explanation flow calls straight into ``get_report``).
"""

from __future__ import annotations

from sqlalchemy import ColumnElement, or_, select
from sqlalchemy.orm import Session

from ..core.errors import NotFoundError
from ..core.logging import get_logger
from ..models import Report
from ..schemas.report import ReportCreate, ReportFilters, ReportUpdate

logger = get_logger(__name__)

#: Catalogue of tests the lab offers. Surfaced to the UI for filter dropdowns.
TEST_TYPES: list[str] = [
    "Blood Test",
    "Sugar Test",
    "Cholesterol",
    "Thyroid",
    "CBC Panel",
    "Urine Test",
    "X-Ray",
    "MRI Scan",
    "Liver Function",
    "Kidney Function",
]

#: Sentinel used by the UI to mean "do not filter on this field".
_ANY = "all"


def list_reports(db: Session, filters: ReportFilters) -> list[Report]:
    """Return reports matching ``filters``, newest first."""
    stmt = select(Report)

    if filters.search:
        like = f"%{filters.search.strip()}%"
        search_terms: list[ColumnElement[bool]] = [
            Report.patient_name.ilike(like),
            Report.test_type.ilike(like),
            Report.city.ilike(like),
            Report.lab_branch.ilike(like),
            Report.doctor_name.ilike(like),
        ]
        if filters.search.isdigit():
            search_terms.append(Report.id == int(filters.search))
        stmt = stmt.where(or_(*search_terms))

    for column, value in (
        (Report.status, filters.status),
        (Report.priority, filters.priority),
        (Report.city, filters.city),
        (Report.test_type, filters.test_type),
    ):
        if value and value != _ANY:
            stmt = stmt.where(column == value)

    stmt = (
        stmt.order_by(Report.created_at.desc(), Report.id.desc())
        .limit(filters.limit)
        .offset(filters.offset)
    )
    return list(db.execute(stmt).scalars())


def get_report(db: Session, report_id: int) -> Report:
    """Fetch one report or raise :class:`NotFoundError`."""
    report = db.get(Report, report_id)
    if report is None:
        raise NotFoundError("Report not found", code="REPORT_NOT_FOUND")
    return report


def create_report(db: Session, payload: ReportCreate) -> Report:
    report = Report(**payload.model_dump())
    db.add(report)
    db.commit()
    db.refresh(report)
    logger.info("report_created", extra={"report_id": report.id})
    return report


def update_report(db: Session, report_id: int, payload: ReportUpdate) -> Report:
    report = get_report(db, report_id)
    changed = payload.model_dump(exclude_unset=True)
    for field, value in changed.items():
        setattr(report, field, value)

    db.commit()
    db.refresh(report)
    logger.info(
        "report_updated",
        extra={"report_id": report.id, "fields": sorted(changed)},
    )
    return report


def delete_report(db: Session, report_id: int) -> None:
    report = get_report(db, report_id)
    db.delete(report)
    db.commit()
    logger.info("report_deleted", extra={"report_id": report_id})


def list_test_types() -> list[str]:
    return list(TEST_TYPES)
