"""Semantic index over sanitised report metadata.

Kept rigorously apart from the knowledge index:

* a **separate Qdrant collection** (``healthlab_reports``), so a report can
  never surface as a citation in a knowledge answer;
* only text the backend has already sanitised. This service never receives a
  patient name, phone number, email or free-text note -- the searchable string
  is built from operational fields alone, and the backend owns that boundary.

Search returns report ids. The caller joins them back to full records from its
own database, so the vector store never has to hold anything identifying.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import Settings, get_settings
from ..core.logging import get_logger
from ..schemas.rag import EmbeddedChunk, KnowledgeChunk, SearchHit
from .embeddings import Embedder
from .vector_store import REPORT_INDEXED_FIELDS, VectorStore

logger = get_logger(__name__)

#: Marks every point in the report collection, so a stray filter cannot mix
#: report vectors into a knowledge query.
REPORT_CATEGORY = "report"

#: Payload keys callers may filter on.
FILTERABLE_FIELDS = frozenset(REPORT_INDEXED_FIELDS)


def report_document_id(report_id: int) -> str:
    """Stable document id for a report."""
    return f"report:{report_id}"


@dataclass(slots=True)
class IndexedReport:
    """One report queued for indexing."""

    report_id: int
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ReportSearchResult:
    """A semantic hit on a report."""

    report_id: int
    score: float
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


class ReportIndexer:
    """Indexes and searches sanitised report metadata."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        settings: Settings | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._settings = settings or get_settings()

    @property
    def collection(self) -> str:
        return self._store.report_collection

    # ------------------------------------------------------------- index ---
    def index(self, reports: list[IndexedReport]) -> int:
        """Embed and store report summaries. Returns points written."""
        if not reports:
            return 0

        started = time.perf_counter()
        vectors = self._embedder.embed_documents([item.text for item in reports])

        chunks = [
            EmbeddedChunk(
                chunk=KnowledgeChunk(
                    document_id=report_document_id(item.report_id),
                    chunk_id=report_document_id(item.report_id),
                    chunk_index=0,
                    title=f"Report #{item.report_id}",
                    source="reports",
                    category=REPORT_CATEGORY,
                    section=None,
                    text=item.text,
                    # Only whitelisted operational keys become filters.
                    extra={
                        key: str(value)
                        for key, value in item.metadata.items()
                        if key in FILTERABLE_FIELDS and value
                    },
                ),
                vector=vector,
            )
            for item, vector in zip(reports, vectors, strict=True)
        ]

        written = self._store.upsert_chunks(
            chunks,
            collection=self.collection,
            index_fields=REPORT_INDEXED_FIELDS,
        )
        logger.info(
            "reports_indexed",
            extra={
                "count": written,
                "collection": self.collection,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return written

    def delete(self, report_id: int) -> int:
        """Remove one report from the index."""
        return self._store.delete_document(
            report_document_id(report_id), collection=self.collection
        )

    # ------------------------------------------------------------ search ---
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[ReportSearchResult]:
        """Find reports semantically similar to ``query``."""
        from .retriever import normalise_query

        cleaned = normalise_query(query)
        vector = self._embedder.embed_text(cleaned)

        # The category filter is always applied, so this can only ever return
        # report points even if the collection were misconfigured.
        applied: dict[str, str] = {"category": REPORT_CATEGORY}
        for key, value in (filters or {}).items():
            if key in FILTERABLE_FIELDS and value:
                applied[key] = value

        hits = self._store.search(
            vector,
            top_k=top_k or self._settings.top_k,
            collection=self.collection,
            score_threshold=score_threshold
            if score_threshold is not None
            else self._settings.score_threshold,
            filters=applied,
        )

        results = [_to_result(hit) for hit in hits]
        logger.info(
            "report_search_completed",
            extra={"hits": len(results), "filtered": len(applied) > 1},
        )
        return [item for item in results if item is not None]

    def stats(self):
        return self._store.stats(self.collection)


def _to_result(hit: SearchHit) -> ReportSearchResult | None:
    """Convert a hit into a report result, or ``None`` if it is not one."""
    raw_id = hit.document_id.removeprefix("report:")
    try:
        report_id = int(raw_id)
    except ValueError:
        logger.warning("report_hit_with_bad_id", extra={"document_id": hit.document_id})
        return None

    return ReportSearchResult(
        report_id=report_id,
        score=hit.score,
        text=hit.text,
        metadata={},
    )
