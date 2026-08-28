"""Schemas for the AI proxy endpoints.

These mirror the AI service's contract explicitly rather than passing raw
dictionaries through, so the backend's OpenAPI document is complete and a
change on the AI side surfaces as a validation error here rather than as a
malformed response reaching the browser.
"""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A knowledge-base source an answer was grounded in."""

    title: str
    source: str
    chunk_id: str
    score: float
    section: str | None = None
    category: str | None = None


class AnswerTimings(BaseModel):
    """Latency breakdown, in milliseconds."""

    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class ChatRequest(BaseModel):
    """Question for the laboratory knowledge assistant."""

    question: str = Field(
        min_length=1,
        max_length=1000,
        description="Natural-language question about tests, terminology or workflow.",
        examples=["What does a CBC test measure?"],
    )
    top_k: int | None = Field(default=None, ge=1, le=20)
    category: str | None = Field(
        default=None, description="Restrict retrieval to one knowledge area."
    )


class ChatResponse(BaseModel):
    """Grounded answer with citations."""

    answer: str
    sources: list[Citation] = Field(default_factory=list)
    retrieval_count: int = 0
    grounded: bool = Field(
        description=(
            "False when nothing relevant was found, when the model declined, "
            "or when the request crossed the clinical boundary. Present the "
            "answer as informational, not authoritative."
        )
    )
    disclaimer: str = Field(description="Healthcare notice to display with the answer.")
    model: str = ""
    provider: str = ""
    finish_reason: str = ""
    timings: AnswerTimings = Field(default_factory=AnswerTimings)


class SearchRequest(BaseModel):
    """Semantic search over the knowledge base."""

    query: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    category: str | None = None


class SearchHit(BaseModel):
    """One retrieved knowledge chunk."""

    text: str
    score: float
    source: str
    title: str
    chunk_id: str
    document_id: str = ""
    section: str | None = None
    category: str | None = None


class SearchResponse(BaseModel):
    """Ranked search results."""

    query: str
    results: list[SearchHit] = Field(default_factory=list)
    retrieval_count: int = 0


class AIComponent(BaseModel):
    """Health of one AI-service component."""

    name: str
    state: str
    detail: str | None = None


class AIHealthResponse(BaseModel):
    """AI-service availability as seen from the backend.

    Always returns 200. ``reachable=false`` is a fact about a dependency, not a
    failure of this API -- the reports system is unaffected either way.
    """

    reachable: bool
    status: str = Field(description="``ok``, ``degraded`` or ``unreachable``.")
    detail: str | None = None
    components: list[AIComponent] = Field(default_factory=list)
