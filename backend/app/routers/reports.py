"""Laboratory report endpoints."""

from fastapi import APIRouter, Query, status

from ..core.errors import NOT_FOUND_RESPONSE
from ..dependencies import DbSession
from ..models import Report
from ..schemas.report import (
    ReportCreate,
    ReportFilters,
    ReportRead,
    ReportUpdate,
)
from ..services import report_service

router = APIRouter(prefix="/api", tags=["Reports"])


@router.get(
    "/test-types",
    response_model=list[str],
    summary="List the test catalogue",
)
def get_test_types() -> list[str]:
    return report_service.list_test_types()


@router.get(
    "/reports",
    response_model=list[ReportRead],
    summary="List reports",
    description="Search and filter diagnostic reports. Pass `all` to a filter "
    "to disable it.",
)
def list_reports(
    db: DbSession,
    search: str | None = Query(default=None, max_length=120),
    status_filter: str | None = Query(default=None, alias="status"),
    priority: str | None = Query(default=None),
    city: str | None = Query(default=None),
    test_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[Report]:
    filters = ReportFilters(
        search=search,
        status=status_filter,
        priority=priority,
        city=city,
        test_type=test_type,
        limit=limit,
        offset=offset,
    )
    return report_service.list_reports(db, filters)


@router.get(
    "/reports/{report_id}",
    response_model=ReportRead,
    responses=NOT_FOUND_RESPONSE,
    summary="Fetch one report",
)
def get_report(report_id: int, db: DbSession) -> Report:
    return report_service.get_report(db, report_id)


@router.post(
    "/reports",
    response_model=ReportRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a report",
)
def create_report(payload: ReportCreate, db: DbSession) -> Report:
    return report_service.create_report(db, payload)


@router.put(
    "/reports/{report_id}",
    response_model=ReportRead,
    responses=NOT_FOUND_RESPONSE,
    summary="Update a report",
    description="Partial update -- only the fields present in the body change.",
)
def update_report(report_id: int, payload: ReportUpdate, db: DbSession) -> Report:
    return report_service.update_report(db, report_id, payload)


@router.delete(
    "/reports/{report_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=NOT_FOUND_RESPONSE,
    summary="Delete a report",
)
def delete_report(report_id: int, db: DbSession) -> None:
    report_service.delete_report(db, report_id)
