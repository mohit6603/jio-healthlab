"""Laboratory report endpoints."""

import time
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Query, status

from ..core.errors import ERROR_RESPONSES, NOT_FOUND_RESPONSE, AppError
from ..core.logging import get_logger
from ..dependencies import CurrentUser, DbSession, Requires
from ..models import Report
from ..schemas.ai import Citation, ExplainRequest, ReportExplanation
from ..schemas.report import (
    ReportCreate,
    ReportFilters,
    ReportRead,
    ReportUpdate,
)
from ..security import Permission
from ..services import audit_service, report_service
from ..services.ai_client import AIServiceClient, get_ai_client
from ..services.audit_service import Action, QueryType, Status
from ..services.sanitizer import sanitize_report

logger = get_logger(__name__)

AIClientDep = Annotated[AIServiceClient, Depends(get_ai_client)]

router = APIRouter(prefix="/api", tags=["Reports"])


@router.get(
    "/test-types",
    response_model=list[str],
    dependencies=[Requires(Permission.REPORTS_READ)],
    summary="List the test catalogue",
)
def get_test_types() -> list[str]:
    return report_service.list_test_types()


@router.get(
    "/reports",
    response_model=list[ReportRead],
    dependencies=[Requires(Permission.REPORTS_READ)],
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
    dependencies=[Requires(Permission.REPORTS_READ)],
    summary="Fetch one report",
)
def get_report(report_id: int, db: DbSession) -> Report:
    return report_service.get_report(db, report_id)


@router.post(
    "/reports",
    response_model=ReportRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Requires(Permission.REPORTS_CREATE)],
    summary="Create a report",
)
def create_report(
    payload: ReportCreate, db: DbSession, user: CurrentUser
) -> Report:
    report = report_service.create_report(db, payload)
    audit_service.record(
        db,
        action=Action.REPORT_CREATED,
        user=user,
        resource_type="report",
        resource_id=report.id,
        detail=f"test_type={report.test_type} priority={report.priority}",
    )
    return report


@router.put(
    "/reports/{report_id}",
    response_model=ReportRead,
    responses=NOT_FOUND_RESPONSE,
    dependencies=[Requires(Permission.REPORTS_UPDATE)],
    summary="Update a report",
    description="Partial update -- only the fields present in the body change.",
)
def update_report(
    report_id: int, payload: ReportUpdate, db: DbSession, user: CurrentUser
) -> Report:
    changed = sorted(payload.model_dump(exclude_unset=True))
    report = report_service.update_report(db, report_id, payload)
    audit_service.record(
        db,
        action=Action.REPORT_UPDATED,
        user=user,
        resource_type="report",
        resource_id=report_id,
        # Field names only. Values could include a patient name or a note.
        detail=f"fields={','.join(changed)}",
    )
    return report


@router.delete(
    "/reports/{report_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=NOT_FOUND_RESPONSE,
    dependencies=[Requires(Permission.REPORTS_DELETE)],
    summary="Delete a report",
)
def delete_report(report_id: int, db: DbSession, user: CurrentUser) -> None:
    report_service.delete_report(db, report_id)
    audit_service.record(
        db,
        action=Action.REPORT_DELETED,
        user=user,
        resource_type="report",
        resource_id=report_id,
    )


@router.post(
    "/reports/{report_id}/explain",
    response_model=ReportExplanation,
    tags=["AI"],
    responses={**ERROR_RESPONSES, **NOT_FOUND_RESPONSE},
    dependencies=[Requires(Permission.AI_EXPLAIN_REPORT)],
    summary="Explain what a report's test measures",
    description=(
        "Explains the requested test in general terms, grounded in the "
        "laboratory knowledge base.\n\n"
        "**No patient identifier is sent.** The report is reduced to a "
        "non-identifying summary first -- test type, status, priority, city, "
        "branch and a ten-year age band. Patient name, phone, email, doctor "
        "name and free-text notes never leave this service. The exact payload "
        "that was sent is returned as `context_sent` so the boundary is "
        "auditable from the response.\n\n"
        "The explanation is informational and contains no diagnosis or "
        "interpretation of this individual's results."
    ),
)
async def explain_report(
    report_id: int,
    db: DbSession,
    client: AIClientDep,
    user: CurrentUser,
    payload: ExplainRequest = Body(default_factory=ExplainRequest),
) -> ReportExplanation:
    report = report_service.get_report(db, report_id)
    sanitized = sanitize_report(report)

    logger.info(
        "report_explanation_requested",
        extra={
            "report_id": report_id,
            "test_type": report.test_type,
            "fields_sent": sorted(sanitized.fields),
        },
    )

    started = time.perf_counter()
    try:
        result = await client.explain_report(
            sanitized.render(),
            search_text=sanitized.search_text,
            top_k=payload.top_k,
        )
    except AppError as exc:
        audit_service.record_ai_query(
            db,
            query_type=QueryType.EXPLAIN,
            user=user,
            success=False,
            error_code=exc.code,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        audit_service.record(
            db,
            action=Action.AI_REPORT_EXPLAINED,
            status=Status.FAILURE,
            user=user,
            resource_type="report",
            resource_id=report_id,
            detail=exc.code,
        )
        raise

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    audit_service.record_ai_query(
        db,
        query_type=QueryType.EXPLAIN,
        user=user,
        latency_ms=latency_ms,
        source_count=result.get("retrieval_count"),
        grounded=result.get("grounded"),
        model=result.get("model"),
    )
    audit_service.record(
        db,
        action=Action.AI_REPORT_EXPLAINED,
        user=user,
        resource_type="report",
        resource_id=report_id,
        latency_ms=latency_ms,
    )

    return ReportExplanation(
        report_id=report_id,
        test_type=report.test_type,
        answer=result.get("answer", ""),
        sources=[Citation(**source) for source in result.get("sources", [])],
        retrieval_count=result.get("retrieval_count", 0),
        grounded=result.get("grounded", False),
        disclaimer=result.get("disclaimer", ""),
        model=result.get("model", ""),
        provider=result.get("provider", ""),
        finish_reason=result.get("finish_reason", ""),
        timings=result.get("timings", {}),
        context_sent=sanitized.fields,
    )
