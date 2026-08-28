"""AI proxy endpoints.

The browser talks to this API, never to the AI service directly. That keeps the
AI service internal, gives one place to apply auth and rate limits later, and
means an AI outage surfaces as a typed error on an endpoint the frontend
already knows how to handle.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from ..core.errors import ERROR_RESPONSES
from ..schemas.ai import (
    AIHealthResponse,
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
)
from ..services.ai_client import AIServiceClient, get_ai_client

router = APIRouter(prefix="/api/ai", tags=["AI"], responses=ERROR_RESPONSES)

AIClientDep = Annotated[AIServiceClient, Depends(get_ai_client)]


@router.get(
    "/health",
    response_model=AIHealthResponse,
    summary="AI service availability",
    description=(
        "Always returns 200. `reachable=false` reports that the AI dependency "
        "is down; reports and the dashboard are unaffected, so this is not "
        "modelled as an error."
    ),
)
async def ai_health(client: AIClientDep) -> AIHealthResponse:
    return AIHealthResponse(**await client.health())


@router.post(
    "/chat",
    response_model=ChatResponse,
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
async def chat(payload: ChatRequest, client: AIClientDep) -> ChatResponse:
    result = await client.chat(
        payload.question, top_k=payload.top_k, category=payload.category
    )
    return ChatResponse(**result)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search over the knowledge base",
    description=(
        "Retrieval only -- no generation. Works even when the language model "
        "is unavailable, so it is a useful fallback for the assistant UI."
    ),
)
async def search(payload: SearchRequest, client: AIClientDep) -> SearchResponse:
    result = await client.search(
        payload.query, top_k=payload.top_k, category=payload.category
    )
    return SearchResponse(**result)
