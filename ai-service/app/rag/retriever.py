"""Semantic retrieval.

    question -> embedding -> Qdrant -> top-k chunks -> ranked results

Deliberately knows nothing about generation. The RAG pipeline composes this
with an LLM; semantic search is also useful on its own, and keeping the two
apart means retrieval can be evaluated -- and can fail -- independently of any
model that writes prose.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings, get_settings
from ..core.errors import AIError
from ..core.logging import get_logger
from ..schemas.rag import SearchHit
from .embeddings import Embedder
from .vector_store import VectorStore

logger = get_logger(__name__)

#: Queries longer than this are rejected. The embedding model truncates at 256
#: word pieces anyway, so anything beyond this is wasted work at best.
MAX_QUERY_CHARS = 1000


class EmptyQueryError(AIError):
    """Raised when a query is blank once whitespace is stripped."""

    status_code = 422
    code = "EMPTY_QUERY"
    message = "The query must contain at least one non-whitespace character."


@dataclass(slots=True)
class RetrievalResult:
    """Ranked chunks for one query, plus the timings behind them."""

    query: str
    hits: list[SearchHit] = field(default_factory=list)
    top_k: int = 0
    score_threshold: float = 0.0
    embed_ms: float = 0.0
    search_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.hits)

    @property
    def is_empty(self) -> bool:
        return not self.hits

    @property
    def best_score(self) -> float | None:
        return self.hits[0].score if self.hits else None

    def sources(self) -> list[str]:
        """Distinct source filenames, in rank order."""
        seen: dict[str, None] = {}
        for hit in self.hits:
            seen.setdefault(hit.source, None)
        return list(seen)


class Retriever:
    """Turns a natural-language question into ranked knowledge chunks."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        settings: Settings | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
        collection: str | None = None,
    ) -> RetrievalResult:
        """Retrieve the chunks most similar to ``query``.

        An empty result is a normal outcome, not an error: it means the
        knowledge base has nothing relevant, which callers must be able to
        report honestly rather than paper over.
        """
        cleaned = normalise_query(query)
        limit = top_k if top_k is not None else self._settings.top_k
        threshold = (
            score_threshold
            if score_threshold is not None
            else self._settings.score_threshold
        )

        started = time.perf_counter()
        vector = self._embedder.embed_text(cleaned)
        embed_ms = round((time.perf_counter() - started) * 1000, 2)

        search_started = time.perf_counter()
        hits = self._store.search(
            vector,
            top_k=limit,
            collection=collection,
            score_threshold=threshold,
            filters=filters,
        )
        search_ms = round((time.perf_counter() - search_started) * 1000, 2)
        total_ms = round((time.perf_counter() - started) * 1000, 2)

        # Qdrant returns best-first, but ranking is a promise this class makes.
        hits.sort(key=lambda hit: hit.score, reverse=True)

        logger.info(
            "retrieval_completed",
            extra={
                "hits": len(hits),
                "top_k": limit,
                "score_threshold": threshold,
                "best_score": hits[0].score if hits else None,
                "embed_ms": embed_ms,
                "search_ms": search_ms,
                "total_ms": total_ms,
                "filtered": bool(filters),
            },
        )

        return RetrievalResult(
            query=cleaned,
            hits=hits,
            top_k=limit,
            score_threshold=threshold,
            embed_ms=embed_ms,
            search_ms=search_ms,
            total_ms=total_ms,
        )


def normalise_query(query: str) -> str:
    """Trim and validate a user query."""
    cleaned = " ".join(query.split())
    if not cleaned:
        raise EmptyQueryError()
    if len(cleaned) > MAX_QUERY_CHARS:
        raise AIError(
            f"The query is {len(cleaned)} characters; the limit is "
            f"{MAX_QUERY_CHARS}.",
            code="QUERY_TOO_LONG",
            status_code=422,
            details={"max_chars": MAX_QUERY_CHARS},
        )
    return cleaned


def build_filters(
    category: str | None = None,
    document_id: str | None = None,
    source: str | None = None,
) -> dict[str, Any] | None:
    """Assemble a metadata filter from the supported request fields."""
    filters = {
        "category": category,
        "document_id": document_id,
        "source": source,
    }
    active = {key: value for key, value in filters.items() if value}
    return active or None
