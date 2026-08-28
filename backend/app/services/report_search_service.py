"""Semantic report search.

The vector index holds only sanitised summaries. Search returns report ids,
and this service joins them back to the full records in MySQL -- so the user
sees a real report while nothing identifying was ever embedded or stored in
the vector database.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.logging import get_logger
from ..models import Report
from .sanitizer import search_metadata, searchable_text

logger = get_logger(__name__)


def index_payload(report: Report) -> dict[str, Any]:
    """The sanitised document for one report."""
    return {
        "report_id": report.id,
        "text": searchable_text(report),
        "metadata": search_metadata(report),
    }


def reports_to_index(db: Session, limit: int) -> list[Report]:
    """Reports to (re)index, newest first."""
    return list(
        db.execute(select(Report).order_by(Report.id.desc()).limit(limit)).scalars()
    )


def hydrate(db: Session, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join semantic hits back to full report records, preserving rank order.

    A hit whose report no longer exists is dropped rather than returned as a
    hollow row -- the index can lag deletions.
    """
    ids = [int(hit["report_id"]) for hit in hits if "report_id" in hit]
    if not ids:
        return []

    found = {
        report.id: report
        for report in db.execute(select(Report).where(Report.id.in_(ids))).scalars()
    }

    stale = [report_id for report_id in ids if report_id not in found]
    if stale:
        logger.info("report_search_stale_hits", extra={"count": len(stale)})

    rows: list[dict[str, Any]] = []
    for hit in hits:
        report = found.get(int(hit["report_id"]))
        if report is None:
            continue
        rows.append(
            {
                "report": report,
                "score": float(hit.get("score", 0.0)),
                "matched_summary": str(hit.get("text", "")),
            }
        )
    return rows
