"""Qdrant vector store.

Wraps every vector operation the service needs so the rest of the codebase
never imports ``qdrant_client`` directly. Connection failures surface as
:class:`VectorStoreError` (HTTP 503) rather than leaking a driver exception.

Two collections are kept deliberately separate:

``healthlab_knowledge``
    Chunks of curated lab documentation -- safe to cite verbatim.
``healthlab_reports``
    Sanitised, non-identifying report metadata for semantic operational search.

Mixing them would let a patient-adjacent record surface as a "source" in a
knowledge answer, so they never share a namespace.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ..config import Settings, get_settings
from ..core.errors import VectorStoreError
from ..core.logging import get_logger
from ..schemas.rag import (
    CollectionStats,
    DocumentSummary,
    EmbeddedChunk,
    SearchHit,
    VectorStoreHealth,
)

logger = get_logger(__name__)

#: Namespace for deterministic point ids. Re-ingesting a document therefore
#: overwrites its existing points instead of duplicating them.
POINT_NAMESPACE = uuid.UUID("6f0b6a3e-6d3a-5c0a-9d2f-2f4a9f0c1e77")

#: Payload fields that get a keyword index so filtering stays fast.
_INDEXED_FIELDS = ("document_id", "category", "source")

#: Extra keyword indexes for the report collection, whose filters are
#: operational rather than bibliographic.
REPORT_INDEXED_FIELDS = ("status", "priority", "branch", "city", "test_type")

_SCROLL_PAGE = 256


def point_id(chunk_id: str) -> str:
    """Derive a deterministic Qdrant point id from a chunk id."""
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


class VectorStore:
    """Thin, testable facade over a Qdrant collection."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._ensured: set[str] = set()

    # ------------------------------------------------------------ client ---
    @property
    def client(self) -> QdrantClient:
        """Create the client on first use so import never opens a socket."""
        if self._client is None:
            try:
                self._client = QdrantClient(
                    url=self._settings.qdrant_url,
                    api_key=self._settings.qdrant_api_key,
                    timeout=int(self._settings.qdrant_timeout_seconds),
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Could not connect to Qdrant at {self._settings.qdrant_url}."
                ) from exc
        return self._client

    @property
    def knowledge_collection(self) -> str:
        return self._settings.qdrant_collection

    @property
    def report_collection(self) -> str:
        return self._settings.qdrant_report_collection

    def _resolve(self, collection: str | None) -> str:
        return collection or self.knowledge_collection

    # -------------------------------------------------------- collection ---
    def ensure_collection(
        self, collection: str | None = None, *, index_fields: tuple[str, ...] = ()
    ) -> str:
        """Create the collection and its payload indexes when missing.

        Idempotent, and memoised per process so the hot path does not pay for
        an existence check on every call.
        """
        name = self._resolve(collection)
        if name in self._ensured:
            return name

        try:
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=qmodels.VectorParams(
                        size=self._settings.embedding_dimension,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
                logger.info(
                    "collection_created",
                    extra={
                        "collection": name,
                        "dimension": self._settings.embedding_dimension,
                    },
                )
            self._ensure_indexes(name, index_fields)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Could not prepare Qdrant collection '{name}'."
            ) from exc

        self._ensured.add(name)
        return name

    def _ensure_indexes(
        self, collection: str, extra_fields: tuple[str, ...] = ()
    ) -> None:
        """Add keyword indexes used by document filters and deletes."""
        for field in (*_INDEXED_FIELDS, *extra_fields):
            try:
                self.client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            # Re-creating an existing index is not an error worth failing on,
            # and the driver signals it with a generic error.
            except Exception as exc:  # noqa: BLE001 - deliberately swallowed
                logger.debug(
                    "payload_index_skipped",
                    extra={"collection": collection, "field": field, "reason": str(exc)},
                )

    def drop_collection(self, collection: str | None = None) -> bool:
        """Delete a collection outright. Used by tests and re-index tooling."""
        name = self._resolve(collection)
        try:
            deleted = bool(self.client.delete_collection(collection_name=name))
        except Exception as exc:
            raise VectorStoreError(f"Could not drop collection '{name}'.") from exc
        self._ensured.discard(name)
        return deleted

    # ------------------------------------------------------------ upsert ---
    def upsert_chunks(
        self,
        chunks: Sequence[EmbeddedChunk],
        collection: str | None = None,
        *,
        index_fields: tuple[str, ...] = (),
    ) -> int:
        """Insert or replace ``chunks``. Returns the number of points written."""
        if not chunks:
            return 0

        name = self.ensure_collection(collection, index_fields=index_fields)
        points = [
            qmodels.PointStruct(
                id=point_id(item.chunk.chunk_id),
                vector=item.vector,
                payload=item.chunk.to_payload(),
            )
            for item in chunks
        ]

        try:
            self.client.upsert(collection_name=name, points=points, wait=True)
        except Exception as exc:
            raise VectorStoreError(
                f"Could not write {len(points)} vectors to '{name}'."
            ) from exc

        logger.info(
            "vectors_upserted", extra={"collection": name, "count": len(points)}
        )
        return len(points)

    # ------------------------------------------------------------ search ---
    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int | None = None,
        collection: str | None = None,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Return the nearest chunks to ``vector``, best first."""
        name = self._resolve(collection)
        limit = top_k or self._settings.top_k

        try:
            response = self.client.query_points(
                collection_name=name,
                query=list(vector),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=_build_filter(filters),
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Vector search against '{name}' failed."
            ) from exc

        hits = [_to_hit(point) for point in response.points]
        logger.info(
            "vector_search_completed",
            extra={"collection": name, "limit": limit, "hits": len(hits)},
        )
        return hits

    # ------------------------------------------------------------ delete ---
    def delete_document(
        self, document_id: str, collection: str | None = None
    ) -> int:
        """Remove every chunk belonging to ``document_id``.

        Returns the number of points that existed before deletion, so callers
        can distinguish "removed 12" from "nothing was indexed".
        """
        name = self._resolve(collection)
        selector = _build_filter({"document_id": document_id})
        assert selector is not None  # a document_id filter always builds

        try:
            # Deleting from a collection that was never created is not an
            # error -- nothing was indexed, so nothing needs removing.
            if not self.client.collection_exists(name):
                return 0
            existing = self.client.count(
                collection_name=name, count_filter=selector, exact=True
            ).count
            if existing:
                self.client.delete(
                    collection_name=name,
                    points_selector=qmodels.FilterSelector(filter=selector),
                    wait=True,
                )
        except Exception as exc:
            raise VectorStoreError(
                f"Could not delete document '{document_id}' from '{name}'."
            ) from exc

        logger.info(
            "document_vectors_deleted",
            extra={"collection": name, "document_id": document_id, "count": existing},
        )
        return int(existing)

    # ------------------------------------------------------------ browse ---
    def list_documents(self, collection: str | None = None) -> list[DocumentSummary]:
        """Summarise indexed documents by scrolling payloads (no vectors)."""
        name = self._resolve(collection)
        summaries: dict[str, DocumentSummary] = {}

        try:
            for payload in self._scroll_payloads(name):
                document_id = str(payload.get("document_id", ""))
                if not document_id:
                    continue
                existing = summaries.get(document_id)
                if existing is None:
                    summaries[document_id] = DocumentSummary(
                        document_id=document_id,
                        title=str(payload.get("title", document_id)),
                        source=str(payload.get("source", "")),
                        category=str(payload.get("category", "")),
                        chunk_count=1,
                    )
                else:
                    existing.chunk_count += 1
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Could not list documents in '{name}'.") from exc

        return sorted(summaries.values(), key=lambda item: item.title.lower())

    def _scroll_payloads(self, collection: str) -> Iterable[dict[str, Any]]:
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                collection_name=collection,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                yield point.payload or {}
            if offset is None:
                return

    def stats(self, collection: str | None = None) -> CollectionStats:
        """Counters for one collection; never raises for a missing collection."""
        name = self._resolve(collection)
        try:
            if not self.client.collection_exists(name):
                return CollectionStats(collection=name, exists=False)
            vector_count = self.client.count(collection_name=name, exact=True).count
            documents = {
                payload.get("document_id")
                for payload in self._scroll_payloads(name)
                if payload.get("document_id")
            }
        except Exception as exc:
            raise VectorStoreError(f"Could not read stats for '{name}'.") from exc

        return CollectionStats(
            collection=name,
            exists=True,
            vector_count=int(vector_count),
            document_count=len(documents),
            dimension=self._settings.embedding_dimension,
        )

    # ------------------------------------------------------------ health ---
    def health(self, collection: str | None = None) -> VectorStoreHealth:
        """Probe reachability. Returns a report instead of raising."""
        name = self._resolve(collection)
        try:
            self.client.get_collections()
            exists = self.client.collection_exists(name)
            count = (
                int(self.client.count(collection_name=name, exact=True).count)
                if exists
                else 0
            )
        # A health probe reports failure, it never propagates it.
        except Exception as exc:  # noqa: BLE001 - deliberately swallowed
            logger.warning("vector_store_unreachable", extra={"collection": name})
            return VectorStoreHealth(
                reachable=False,
                collection=name,
                detail=f"{type(exc).__name__}: {exc}",
                checked_at=datetime.now(UTC),
            )

        return VectorStoreHealth(
            reachable=True,
            collection=name,
            collection_exists=exists,
            vector_count=count,
            checked_at=datetime.now(UTC),
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._ensured.clear()


# --------------------------------------------------------------- helpers ---
def _build_filter(filters: dict[str, Any] | None) -> qmodels.Filter | None:
    """Translate ``{field: value}`` (or ``{field: [v1, v2]}``) into a Filter."""
    if not filters:
        return None

    conditions: list[qmodels.FieldCondition] = []
    for field, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values = [str(item) for item in value]
            if not values:
                continue
            conditions.append(
                qmodels.FieldCondition(
                    key=field, match=qmodels.MatchAny(any=values)
                )
            )
        else:
            conditions.append(
                qmodels.FieldCondition(
                    key=field, match=qmodels.MatchValue(value=value)
                )
            )

    return qmodels.Filter(must=conditions) if conditions else None


def _to_hit(point: Any) -> SearchHit:
    """Convert a Qdrant scored point into a :class:`SearchHit`."""
    payload = point.payload or {}
    return SearchHit(
        text=str(payload.get("text", "")),
        score=float(point.score),
        source=str(payload.get("source", "")),
        title=str(payload.get("title", "")),
        chunk_id=str(payload.get("chunk_id", point.id)),
        document_id=str(payload.get("document_id", "")),
        category=payload.get("category"),
        section=payload.get("section"),
    )


# --------------------------------------------------------------- factory ---
_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Process-wide singleton -- one Qdrant client, reused across requests."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def reset_vector_store() -> None:
    """Drop the singleton (test helper)."""
    global _store
    if _store is not None:
        _store.close()
    _store = None
