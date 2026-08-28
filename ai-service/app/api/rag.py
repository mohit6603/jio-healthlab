"""Semantic search endpoints.

Retrieval is exposed separately from generation so it can be evaluated,
monitored and used on its own -- and so it keeps working when no generation
model is available.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..core.errors import ERROR_RESPONSES
from ..rag.retriever import Retriever, build_filters
from ..schemas.rag import RetrievalTimings, SearchRequest, SearchResponse
from .deps import EmbedderDep, SettingsDep, VectorStoreDep

router = APIRouter(prefix="/rag", tags=["RAG"], responses=ERROR_RESPONSES)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search over the knowledge base",
    description=(
        "Embeds the query and returns the most similar knowledge chunks, "
        "best first, with the citation metadata needed to attribute them.\n\n"
        "An empty `results` list is a normal outcome: it means nothing in the "
        "knowledge base is relevant above the score threshold. No generation "
        "model is involved, so this endpoint works even when the LLM is "
        "unavailable."
    ),
)
def search(
    payload: SearchRequest,
    settings: SettingsDep,
    store: VectorStoreDep,
    embedder: EmbedderDep,
) -> SearchResponse:
    retriever = Retriever(embedder, store, settings)
    result = retriever.retrieve(
        payload.query,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
        filters=build_filters(
            category=payload.category,
            document_id=payload.document_id,
            source=payload.source,
        ),
    )

    return SearchResponse(
        query=result.query,
        results=result.hits,
        retrieval_count=result.count,
        top_k=result.top_k,
        score_threshold=result.score_threshold,
        timings=RetrievalTimings(
            embed_ms=result.embed_ms,
            search_ms=result.search_ms,
            total_ms=result.total_ms,
        ),
    )
