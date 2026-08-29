"""Semantic search endpoints.

Retrieval is exposed separately from generation so it can be evaluated,
monitored and used on its own -- and so it keeps working when no generation
model is available.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..core.errors import ERROR_RESPONSES, AIError
from ..rag.pipeline import RagAnswer, RagPipeline
from ..rag.prompts import AI_DISCLAIMER
from ..rag.retriever import Retriever, build_filters, normalise_query
from ..schemas.rag import (
    AnswerTimings,
    Citation,
    ExplainRequest,
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


@router.post(
    "/explain",
    response_model=QueryResponse,
    summary="Explain a laboratory request in general terms",
    description=(
        "Explains what the requested test measures and what its terminology "
        "means, grounded in the knowledge base.\n\n"
        "`report_summary` **must already be sanitised** -- this service never "
        "receives patient identifiers. The prompt instructs the model to "
        "describe the test in general terms only and to state no finding or "
        "interpretation for the individual, and the same clinical screening "
        "applied to `/rag/query` applies here."
    ),
)
def explain(
    payload: ExplainRequest,
    settings: SettingsDep,
    store: VectorStoreDep,
    embedder: EmbedderDep,
    provider: LLMProviderDep,
) -> QueryResponse:
    pipeline = RagPipeline(Retriever(embedder, store, settings), provider, settings)
    result = pipeline.explain_report(
        payload.report_summary,
        search_text=payload.search_text,
        top_k=payload.top_k,
        max_new_tokens=payload.max_new_tokens,
    )
    return to_query_response(result)


def _sse(event: str, payload: dict) -> str:
    """Render one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@router.post(
    "/query/stream",
    summary="Ask a grounded question, streamed",
    description=(
        "Same pipeline as `POST /rag/query`, delivered as Server-Sent Events "
        "so the answer appears as it is written rather than after 10-30 "
        "seconds of silence.\n\n"
        "**Event order** — `sources` first, so the UI can show what the answer "
        "is grounded in while it is still being written; then `token` events; "
        "then exactly one `done`.\n\n"
        "A `replace` event means the safety screen rejected the completed "
        "text: the client must discard what it has rendered and show the "
        "replacement instead. That check can only run on the finished answer, "
        "which is the one cost of streaming.\n\n"
        "An `error` event carries the same codes as the non-streaming "
        "endpoint. Errors arrive as events rather than status codes because "
        "the response has already begun."
    ),
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
def query_stream(
    payload: QueryRequest,
    settings: SettingsDep,
    store: VectorStoreDep,
    embedder: EmbedderDep,
    provider: LLMProviderDep,
) -> StreamingResponse:
    # Validate before the response begins. Once streaming starts the status
    # code is fixed at 200, so a bad request could only be reported as an
    # event -- which clients would have to special-case. Fail normally here.
    question = normalise_query(payload.question)

    pipeline = RagPipeline(Retriever(embedder, store, settings), provider, settings)

    def events() -> Iterator[str]:
        try:
            for event in pipeline.stream_answer(
                question,
                top_k=payload.top_k,
                score_threshold=payload.score_threshold,
                category=payload.category,
                max_new_tokens=payload.max_new_tokens,
            ):
                if event.type == "sources":
                    yield _sse(
                        "sources",
                        {
                            "sources": [
                                Citation(
                                    title=hit.title,
                                    source=hit.source,
                                    chunk_id=hit.chunk_id,
                                    score=hit.score,
                                    section=hit.section,
                                    category=hit.category,
                                ).model_dump()
                                for hit in event.hits
                            ]
                        },
                    )
                elif event.type in {"token", "replace"}:
                    yield _sse(event.type, {"text": event.text})
                elif event.type == "done":
                    yield _sse(
                        "done",
                        {
                            "grounded": event.grounded,
                            "finish_reason": event.finish_reason,
                            "model": event.model,
                            "provider": event.provider,
                            "retrieval_count": event.retrieval_count,
                            "drop_sources": event.drop_sources,
                            "disclaimer": AI_DISCLAIMER,
                            "timings": {
                                "retrieval_ms": event.retrieval_ms,
                                "generation_ms": 0.0,
                                "total_ms": event.total_ms,
                            },
                        },
                    )
        except AIError as exc:
            # The response has already started, so a status code is no longer
            # available; the client reads the code from the event instead.
            yield _sse("error", {"code": exc.code, "message": exc.message})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Tells nginx not to buffer, which would defeat streaming entirely.
            "X-Accel-Buffering": "no",
        },
    )
