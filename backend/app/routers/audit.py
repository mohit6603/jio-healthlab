"""Audit trail endpoints.

Administrators only: the trail records who did what, and exposing it more
widely would leak the shape of other people's activity.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..core.errors import ERROR_RESPONSES
from ..dependencies import DbSession, Requires
from ..schemas.audit import AIQueryLogRead, AIUsageSummary, AuditLogRead
from ..security import Permission
from ..services import audit_service

router = APIRouter(
    prefix="/api/audit",
    tags=["Audit"],
    responses=ERROR_RESPONSES,
    dependencies=[Requires(Permission.ADMIN_USERS)],
)


@router.get(
    "",
    response_model=list[AuditLogRead],
    summary="Audit trail",
    description=(
        "Recorded actions, most recent first. The trail is append-only: there "
        "is no endpoint to modify or delete an entry.\n\n"
        "`detail` carries identifiers and enum-like values only -- never a "
        "patient identifier, a free-text note, or an AI prompt."
    ),
)
def list_audit_logs(
    db: DbSession,
    action: str | None = Query(default=None, description="Exact action name."),
    status: str | None = Query(
        default=None, description="`success`, `failure` or `denied`."
    ),
    user_id: int | None = Query(default=None),
    resource_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[AuditLogRead]:
    entries = audit_service.list_audit_logs(
        db,
        action=action,
        status=status,
        user_id=user_id,
        resource_id=resource_id,
        limit=limit,
        offset=offset,
    )
    return [AuditLogRead.model_validate(entry) for entry in entries]


@router.get(
    "/ai-queries",
    response_model=list[AIQueryLogRead],
    summary="AI usage log",
    description=(
        "One row per AI interaction, most recent first.\n\n"
        "**No question, answer or retrieved text is stored.** Assistant "
        "questions can be patient-adjacent, and keeping them would create a "
        "second copy of sensitive text outside the reports table for no "
        "operational gain the metadata does not already provide."
    ),
)
def list_ai_query_logs(
    db: DbSession,
    query_type: str | None = Query(default=None),
    user_id: int | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[AIQueryLogRead]:
    entries = audit_service.list_ai_query_logs(
        db, query_type=query_type, user_id=user_id, limit=limit, offset=offset
    )
    return [AIQueryLogRead.model_validate(entry) for entry in entries]


@router.get(
    "/ai-usage",
    response_model=AIUsageSummary,
    summary="Aggregate AI usage",
    description="Call counts, failure counts and average latency by query type.",
)
def ai_usage(db: DbSession) -> AIUsageSummary:
    return AIUsageSummary(**audit_service.ai_usage_summary(db))
