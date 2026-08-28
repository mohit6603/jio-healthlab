"""Recording audit and AI-usage events.

Writes are best-effort. An audit insert failing must not turn a successful
report update into a 500 for the user -- the failure is logged loudly instead,
where alerting can pick it up. Actions are recorded *after* the operation
succeeds or fails, so the row reflects what actually happened.

Nothing here accepts free text from a user. ``detail`` is built from
identifiers and enum-like values by the caller.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.logging import get_logger
from ..core.request_context import get_request_id
from ..models import AIQueryLog, AuditLog, User

logger = get_logger(__name__)


class Action(StrEnum):
    """Auditable actions. One value per thing worth answering questions about."""

    LOGIN_SUCCEEDED = "login.succeeded"
    LOGIN_FAILED = "login.failed"
    LOGOUT = "logout"
    SESSION_REFRESHED = "session.refreshed"
    SESSION_REUSE_DETECTED = "session.reuse_detected"

    USER_CREATED = "user.created"

    REPORT_CREATED = "report.created"
    REPORT_UPDATED = "report.updated"
    REPORT_DELETED = "report.deleted"

    AI_CHAT = "ai.chat"
    AI_SEARCH = "ai.search"
    AI_REPORT_EXPLAINED = "ai.report_explained"
    AI_REPORT_SEARCH = "ai.report_search"
    AI_RISK_ANALYTICS = "ai.risk_analytics"

    DOCUMENT_INGESTED = "document.ingested"
    DOCUMENT_DELETED = "document.deleted"
    REPORT_INDEX_REBUILT = "report_index.rebuilt"

    PERMISSION_DENIED = "permission.denied"


class Status(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"


class QueryType(StrEnum):
    CHAT = "chat"
    SEARCH = "search"
    EXPLAIN = "explain"
    REPORT_SEARCH = "report_search"
    RISK_ANALYTICS = "risk_analytics"


def record(
    db: Session,
    *,
    action: Action | str,
    status: Status | str = Status.SUCCESS,
    user: User | None = None,
    resource_type: str | None = None,
    resource_id: str | int | None = None,
    detail: str | None = None,
    latency_ms: float | None = None,
) -> AuditLog | None:
    """Append one audit row. Returns None if the write failed."""
    entry = AuditLog(
        user_id=user.id if user else None,
        user_role=user.role if user else None,
        action=str(action),
        status=str(status),
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id is not None else None,
        detail=detail,
        request_id=get_request_id(),
        latency_ms=latency_ms,
    )

    try:
        db.add(entry)
        db.commit()
        db.refresh(entry)
    except Exception:
        # An unwritable audit trail is serious, but failing the user's request
        # because of it is worse. Log loudly and continue.
        db.rollback()
        logger.exception("audit_write_failed", extra={"action": str(action)})
        return None

    return entry


def record_ai_query(
    db: Session,
    *,
    query_type: QueryType | str,
    user: User | None = None,
    success: bool = True,
    error_code: str | None = None,
    latency_ms: float | None = None,
    source_count: int | None = None,
    grounded: bool | None = None,
    model: str | None = None,
) -> AIQueryLog | None:
    """Append one AI-usage row.

    The question is deliberately not a parameter: there is nowhere to put it.
    """
    entry = AIQueryLog(
        user_id=user.id if user else None,
        query_type=str(query_type),
        success=success,
        error_code=error_code,
        latency_ms=latency_ms,
        source_count=source_count,
        grounded=grounded,
        model=model,
        request_id=get_request_id(),
    )

    try:
        db.add(entry)
        db.commit()
        db.refresh(entry)
    except Exception:
        db.rollback()
        logger.exception("ai_query_log_write_failed")
        return None

    return entry


def list_audit_logs(
    db: Session,
    *,
    action: str | None = None,
    status: str | None = None,
    user_id: int | None = None,
    resource_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[AuditLog]:
    """Most recent first."""
    stmt = select(AuditLog)
    for column, value in (
        (AuditLog.action, action),
        (AuditLog.status, status),
        (AuditLog.resource_id, resource_id),
    ):
        if value:
            stmt = stmt.where(column == value)
    if user_id is not None:
        stmt = stmt.where(AuditLog.user_id == user_id)

    stmt = stmt.order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
    return list(db.execute(stmt.limit(limit).offset(offset)).scalars())


def list_ai_query_logs(
    db: Session,
    *,
    query_type: str | None = None,
    user_id: int | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[AIQueryLog]:
    """Most recent first."""
    stmt = select(AIQueryLog)
    if query_type:
        stmt = stmt.where(AIQueryLog.query_type == query_type)
    if user_id is not None:
        stmt = stmt.where(AIQueryLog.user_id == user_id)

    stmt = stmt.order_by(AIQueryLog.created_at.desc(), AIQueryLog.id.desc())
    return list(db.execute(stmt.limit(limit).offset(offset)).scalars())


def ai_usage_summary(db: Session) -> dict[str, Any]:
    """Counts and latency by query type, for the admin view."""
    # `case` is a SQL construct, not a function: func.case() would emit a
    # literal CASE(...) call and fail on every dialect.
    from sqlalchemy import case, func

    rows = db.execute(
        select(
            AIQueryLog.query_type,
            func.count(AIQueryLog.id),
            func.sum(case((AIQueryLog.success.is_(False), 1), else_=0)),
            func.avg(AIQueryLog.latency_ms),
        ).group_by(AIQueryLog.query_type)
    ).all()

    return {
        "by_type": [
            {
                "query_type": str(query_type),
                "count": int(count or 0),
                "failures": int(failures or 0),
                "average_latency_ms": round(float(latency), 2) if latency else 0.0,
            }
            for query_type, count, failures, latency in rows
        ],
        "total": sum(int(count or 0) for _, count, _, _ in rows),
    }
