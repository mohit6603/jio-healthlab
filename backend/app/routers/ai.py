"""AI proxy endpoints.

The browser talks to this API, never to the AI service directly. That keeps the
AI service internal, gives one place to apply auth and rate limits later, and
means an AI outage surfaces as a typed error on an endpoint the frontend
already knows how to handle.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from ..core.errors import ERROR_RESPONSES, AppError
from ..dependencies import CurrentUser, DbSession, Requires
from ..schemas.ai import (
    AIHealthResponse,
    ChatRequest,
    ChatResponse,
    ReportIndexResponse,
    ReportSearchMatch,
    ReportSearchRequest,
    ReportSearchResponse,
    RiskAnalyticsResponse,
    SearchRequest,
    SearchResponse,
)
from ..schemas.report import ReportRead
from ..security import Permission
from ..services import audit_service, report_search_service, risk_service
from ..services.ai_client import AIServiceClient, get_ai_client
from ..services.audit_service import Action, QueryType

router = APIRouter(prefix="/api/ai", tags=["AI"], responses=ERROR_RESPONSES)

AIClientDep = Annotated[AIServiceClient, Depends(get_ai_client)]


@router.get(
    "/health",
    response_model=AIHealthResponse,
    summary="AI service availability",
    description=(
        "Requires authentication but no specific permission. Always returns "
        "200: `reachable=false` reports that the AI dependency "
        "is down; reports and the dashboard are unaffected, so this is not "
        "modelled as an error."
    ),
)
async def ai_health(client: AIClientDep, user: CurrentUser) -> AIHealthResponse:
    return AIHealthResponse(**await client.health())


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Requires(Permission.AI_CHAT)],
    summary="Ask the laboratory knowledge assistant",
    description=(
        "Answers from the laboratory knowledge base and returns the sources "
        "used.\n\n"
        "`grounded=false` means the answer is not backed by retrieved "
        "material -- nothing relevant was found, the model declined, or the "
        "question asked for clinical interpretation, which this assistant does "
        "not provide.\n\n"
        "Returns **503** when the AI service or its model is unavailable and "
        "**504** when generation exceeds the timeout."
    ),
)
async def chat(
    payload: ChatRequest, client: AIClientDep, db: DbSession, user: CurrentUser
) -> ChatResponse:
    started = time.perf_counter()
    try:
        result = await client.chat(
            payload.question, top_k=payload.top_k, category=payload.category
        )
    except AppError as exc:
        _log_ai(db, QueryType.CHAT, user, started, error_code=exc.code)
        raise

    # The question itself is never stored -- only how the call went.
    _log_ai(
        db,
        QueryType.CHAT,
        user,
        started,
        source_count=result.get("retrieval_count"),
        grounded=result.get("grounded"),
        model=result.get("model"),
    )
    return ChatResponse(**result)


@router.post(
    "/search",
    response_model=SearchResponse,
    dependencies=[Requires(Permission.AI_SEARCH)],
    summary="Semantic search over the knowledge base",
    description=(
        "Retrieval only -- no generation. Works even when the language model "
        "is unavailable, so it is a useful fallback for the assistant UI."
    ),
)
async def search(
    payload: SearchRequest, client: AIClientDep, db: DbSession, user: CurrentUser
) -> SearchResponse:
    started = time.perf_counter()
    try:
        result = await client.search(
            payload.query, top_k=payload.top_k, category=payload.category
        )
    except AppError as exc:
        _log_ai(db, QueryType.SEARCH, user, started, error_code=exc.code)
        raise

    _log_ai(
        db,
        QueryType.SEARCH,
        user,
        started,
        source_count=result.get("retrieval_count"),
    )
    return SearchResponse(**result)


@router.get(
    "/risk-analytics",
    response_model=RiskAnalyticsResponse,
    dependencies=[Requires(Permission.AI_RISK_ANALYTICS)],
    summary="Predicted delay risk across in-flight reports",
    description=(
        "Scores every report that can still miss its turnaround target and "
        "aggregates the result.\n\n"
        "Queue features are computed from the database -- how many requests "
        "are open at each branch and how many of those are urgent -- so the "
        "scores reflect the lab's actual state rather than supplied "
        "estimates. Reports already `ready` or `delivered` are excluded: they "
        "cannot become late.\n\n"
        "**No patient identifier is sent to the AI service or returned.** "
        "Rows are keyed by report id.\n\n"
        "Returns **503** when the AI service or the model is unavailable; the "
        "reports API is unaffected."
    ),
)
async def risk_analytics(
    db: DbSession,
    client: AIClientDep,
    user: CurrentUser,
    limit: int = Query(
        default=100, ge=1, le=500, description="Maximum reports to score."
    ),
) -> RiskAnalyticsResponse:
    reports = risk_service.in_flight_reports(db, limit)
    if not reports:
        return RiskAnalyticsResponse(**asdict(risk_service.empty_analytics()))

    loads = risk_service.branch_load(db)
    features = [risk_service.build_features(report, loads) for report in reports]

    started = time.perf_counter()
    try:
        result = await client.predict_delay_batch(features)
    except AppError as exc:
        _log_ai(db, QueryType.RISK_ANALYTICS, user, started, error_code=exc.code)
        raise

    predictions = result.get("predictions", [])
    _log_ai(
        db,
        QueryType.RISK_ANALYTICS,
        user,
        started,
        source_count=len(predictions),
        model=str(result.get("model_version", "")),
    )

    analytics = risk_service.aggregate(
        reports,
        predictions,
        model_version=str(result.get("model_version", "")),
        synthetic_model=bool(
            predictions[0].get("synthetic_model", True) if predictions else True
        ),
    )
    return RiskAnalyticsResponse(**asdict(analytics))


@router.post(
    "/report-search",
    response_model=ReportSearchResponse,
    dependencies=[Requires(Permission.AI_REPORT_SEARCH)],
    summary="Search reports in natural language",
    description=(
        "Matches a phrase such as *urgent kidney tests waiting in Mumbai* "
        "against the semantic report index, then returns the full report "
        "records for the matches.\n\n"
        "Only sanitised summaries are embedded -- operational fields, an age "
        "band, no names or contact details. The vector store holds report ids "
        "and that summary; the records themselves come from the database. "
        "Optional exact filters narrow the search further.\n\n"
        "Returns **503** when the AI service or the index is unavailable; "
        "keyword search on `GET /api/reports` is unaffected."
    ),
)
async def report_search(
    payload: ReportSearchRequest,
    db: DbSession,
    client: AIClientDep,
    user: CurrentUser,
) -> ReportSearchResponse:
    started = time.perf_counter()
    result = await client.search_reports(
        payload.query,
        top_k=payload.top_k,
        status=payload.status,
        priority=payload.priority,
        branch=payload.branch,
        city=payload.city,
        test_type=payload.test_type,
    )

    rows = report_search_service.hydrate(db, result.get("results", []))
    _log_ai(db, QueryType.REPORT_SEARCH, user, started, source_count=len(rows))
    return ReportSearchResponse(
        query=payload.query.strip(),
        results=[
            ReportSearchMatch(
                report=ReportRead.model_validate(row["report"]),
                score=row["score"],
                matched_summary=row["matched_summary"],
            )
            for row in rows
        ],
        retrieval_count=len(rows),
    )


@router.post(
    "/report-index",
    response_model=ReportIndexResponse,
    dependencies=[Requires(Permission.ADMIN_INDEX)],
    summary="Rebuild the semantic report index",
    description=(
        "Sanitises reports and pushes their summaries to the AI service. "
        "Re-indexing a report replaces its vector rather than duplicating it, "
        "so this is safe to re-run.\n\n"
        "An administrative operation: it decides what report search can find."
    ),
)
async def rebuild_report_index(
    db: DbSession,
    client: AIClientDep,
    user: CurrentUser,
    limit: int = Query(default=500, ge=1, le=1000),
) -> ReportIndexResponse:
    reports = report_search_service.reports_to_index(db, limit)
    if not reports:
        return ReportIndexResponse(indexed=0, collection="")

    payload = [report_search_service.index_payload(report) for report in reports]
    result = await client.index_reports(payload)
    audit_service.record(
        db,
        action=Action.REPORT_INDEX_REBUILT,
        user=user,
        resource_type="report_index",
        detail=f"indexed={result.get('indexed', 0)}",
    )
    return ReportIndexResponse(
        indexed=int(result.get("indexed", 0)),
        collection=str(result.get("collection", "")),
    )


def _log_ai(
    db,
    query_type: QueryType,
    user,
    started: float,
    *,
    error_code: str | None = None,
    source_count: int | None = None,
    grounded: bool | None = None,
    model: str | None = None,
) -> None:
    """Record AI usage metadata. Never the question, never the answer."""
    audit_service.record_ai_query(
        db,
        query_type=query_type,
        user=user,
        success=error_code is None,
        error_code=error_code,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        source_count=source_count,
        grounded=grounded,
        model=model,
    )
