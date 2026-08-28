"""Semantic search over sanitised report metadata.

A separate collection from the knowledge base, and a separate router, because
the two must never be confused: knowledge chunks are citable reference
material, report vectors are operational metadata about real requests.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from ..core.errors import ERROR_RESPONSES
from ..rag.report_index import IndexedReport, ReportIndexer
from ..schemas.rag import (
    ReportIndexRequest,
    ReportIndexResponse,
    ReportIndexStats,
    ReportSearchHit,
    ReportSearchRequest,
    ReportSearchResponse,
)
from .deps import EmbedderDep, SettingsDep, VectorStoreDep

router = APIRouter(
    prefix="/reports", tags=["Report search"], responses=ERROR_RESPONSES
)

_BOUNDARY_NOTE = (
    "\n\n> Indexed text is built by the backend from operational fields only. "
    "This service never receives patient names, contact details or free-text "
    "notes, and report vectors live in a collection separate from the "
    "knowledge base."
)


@router.post(
    "/index",
    response_model=ReportIndexResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Index sanitised report summaries",
    description=(
        "Embeds and stores report summaries for semantic search. Re-indexing a "
        "report replaces its existing vector rather than duplicating it."
        + _BOUNDARY_NOTE
    ),
)
def index_reports(
    payload: ReportIndexRequest,
    settings: SettingsDep,
    store: VectorStoreDep,
    embedder: EmbedderDep,
) -> ReportIndexResponse:
    indexer = ReportIndexer(embedder, store, settings)
    written = indexer.index(
        [
            IndexedReport(
                report_id=item.report_id, text=item.text, metadata=item.metadata
            )
            for item in payload.items
        ]
    )
    return ReportIndexResponse(indexed=written, collection=indexer.collection)


@router.post(
    "/search",
    response_model=ReportSearchResponse,
    summary="Search reports in natural language",
    description=(
        "Matches a phrase such as *urgent kidney tests waiting in Mumbai* "
        "against indexed report summaries, with optional exact filters on "
        "status, priority, branch, city and test type.\n\n"
        "Returns report ids and the matched summary. The caller joins those "
        "ids back to full records from its own database, so nothing "
        "identifying has to live in the vector store." + _BOUNDARY_NOTE
    ),
)
def search_reports(
    payload: ReportSearchRequest,
    settings: SettingsDep,
    store: VectorStoreDep,
    embedder: EmbedderDep,
) -> ReportSearchResponse:
    indexer = ReportIndexer(embedder, store, settings)
    results = indexer.search(
        payload.query,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
        filters={
            "status": payload.status or "",
            "priority": payload.priority or "",
            "branch": payload.branch or "",
            "city": payload.city or "",
            "test_type": payload.test_type or "",
        },
    )

    return ReportSearchResponse(
        query=payload.query.strip(),
        results=[
            ReportSearchHit(
                report_id=item.report_id, score=item.score, text=item.text
            )
            for item in results
        ],
        retrieval_count=len(results),
    )


@router.delete(
    "/{report_id}",
    response_model=ReportIndexResponse,
    summary="Remove a report from the index",
    description="Deletes the report's vector. Deleting an unindexed report is "
    "not an error.",
)
def delete_report(
    report_id: int,
    settings: SettingsDep,
    store: VectorStoreDep,
    embedder: EmbedderDep,
) -> ReportIndexResponse:
    indexer = ReportIndexer(embedder, store, settings)
    return ReportIndexResponse(
        indexed=-indexer.delete(report_id), collection=indexer.collection
    )


@router.get(
    "/index/stats",
    response_model=ReportIndexStats,
    summary="Report index size",
    description="How many reports are currently searchable, and in which "
    "collection. `exists: false` means nothing has been indexed yet -- run "
    "`POST /api/ai/report-index` on the backend.",
)
def index_stats(
    settings: SettingsDep, store: VectorStoreDep, embedder: EmbedderDep
) -> ReportIndexStats:
    stats = ReportIndexer(embedder, store, settings).stats()
    return ReportIndexStats(
        collection=stats.collection,
        exists=stats.exists,
        indexed_reports=stats.document_count,
    )
