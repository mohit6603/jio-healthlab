"""Semantic search endpoints.

Retrieval is exposed separately from generation so it can be evaluated,
monitored and used on its own -- and so it keeps working when no generation
model is available.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..core.errors import ERROR_RESPONSES
from ..rag.pipeline import RagAnswer, RagPipeline
from ..rag.retriever import Retriever, build_filters
from ..schemas.rag import (
    AnswerTimings,
    Citation,
    QueryRequest,
    QueryResponse,
    RetrievalTimings,
    SearchRequest,
    SearchResponse,
)
from .deps import EmbedderDep, LLMProviderDep, SettingsDep, VectorStoreDep

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


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a grounded question",
    description=(
        "Retrieves relevant knowledge chunks and answers from them, returning "
        "the sources the answer was grounded in.\n\n"
        "When nothing relevant is retrieved the model is **not** called: a "
        "fixed 'not enough information' message is returned with "
        "`grounded=false` and no sources. Returns **503** when generation is "
        "disabled or the model cannot be loaded -- `/rag/search` still works "
        "in that case."
    ),
)
def query(
    payload: QueryRequest,
    settings: SettingsDep,
    store: VectorStoreDep,
    embedder: EmbedderDep,
    provider: LLMProviderDep,
) -> QueryResponse:
    pipeline = RagPipeline(Retriever(embedder, store, settings), provider, settings)
    result = pipeline.answer(
        payload.question,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
        category=payload.category,
        max_new_tokens=payload.max_new_tokens,
    )
    return to_query_response(result)


def to_query_response(result: RagAnswer) -> QueryResponse:
    """Map the pipeline result onto the wire schema."""
    return QueryResponse(
        answer=result.answer,
        sources=[
            Citation(
                title=hit.title,
                source=hit.source,
                chunk_id=hit.chunk_id,
                score=hit.score,
                section=hit.section,
                category=hit.category,
            )
            for hit in result.hits
        ],
        retrieval_count=result.retrieval_count,
        grounded=result.grounded,
        disclaimer=result.disclaimer,
        model=result.model,
        provider=result.provider,
        finish_reason=result.finish_reason,
        prompt_truncated=result.prompt_truncated,
        timings=AnswerTimings(
            retrieval_ms=result.retrieval_ms,
            generation_ms=result.generation_ms,
            total_ms=result.total_ms,
        ),
    )
