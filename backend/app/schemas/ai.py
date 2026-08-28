"""Schemas for the AI proxy endpoints.

These mirror the AI service's contract explicitly rather than passing raw
dictionaries through, so the backend's OpenAPI document is complete and a
change on the AI side surfaces as a validation error here rather than as a
malformed response reaching the browser.
"""

from pydantic import BaseModel, Field

from .report import ReportRead


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


class ExplainRequest(BaseModel):
    """Options for explaining a report. The report itself comes from the path."""

    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Knowledge chunks to retrieve."
    )


class ReportExplanation(BaseModel):
    """AI explanation of what a report's test measures.

    Describes the test in general terms. It contains no finding, result or
    clinical interpretation for the individual, and no patient identifier was
    sent to produce it.
    """

    report_id: int
    test_type: str
    answer: str
    sources: list[Citation] = Field(default_factory=list)
    retrieval_count: int = 0
    grounded: bool
    disclaimer: str
    model: str = ""
    provider: str = ""
    finish_reason: str = ""
    timings: AnswerTimings = Field(default_factory=AnswerTimings)
    #: Exactly what was sent to the AI service, so the boundary is auditable
    #: from the response itself.
    context_sent: dict[str, str] = Field(default_factory=dict)


class RiskGroup(BaseModel):
    """Aggregated risk for one branch or test type."""

    label: str
    count: int
    average_probability: float
    high_risk: int


class HighestRiskBranch(BaseModel):
    branch: str
    average_probability: float
    reports: int
    high_risk: int


class ReportRisk(BaseModel):
    """Predicted delay risk for one in-flight report.

    Deliberately carries no patient identifier -- the row is keyed by report
    id, and everything else is operational metadata.
    """

    report_id: int
    test_type: str
    branch: str | None = None
    city: str | None = None
    priority: str | None = None
    status: str | None = None
    result_due_at: str | None = None
    delay_probability: float
    risk_level: str


class RiskAnalyticsResponse(BaseModel):
    """Operational delay-risk snapshot for the AI analytics dashboard."""

    generated_at: str
    model_version: str = ""
    synthetic_model: bool = Field(
        default=True,
        description=(
            "True when the scoring model was trained on synthetic data. These "
            "figures demonstrate the pipeline; they are not an operational "
            "forecast."
        ),
    )
    reports_scored: int = 0
    at_risk: int = Field(default=0, description="Reports in the medium or high band.")
    high_risk: int = 0
    predicted_late: int = Field(
        default=0, description="Reports with probability at or above 0.5."
    )
    average_probability: float = 0.0
    highest_risk_branch: HighestRiskBranch | None = None
    risk_distribution: dict[str, int] = Field(default_factory=dict)
    by_branch: list[RiskGroup] = Field(default_factory=list)
    by_test_type: list[RiskGroup] = Field(default_factory=list)
    reports: list[ReportRisk] = Field(default_factory=list)


class ReportSearchRequest(BaseModel):
    """Natural-language search over lab reports."""

    query: str = Field(
        min_length=1,
        max_length=1000,
        examples=["urgent kidney tests waiting in Mumbai"],
    )
    top_k: int | None = Field(default=None, ge=1, le=50)
    status: str | None = None
    priority: str | None = None
    branch: str | None = None
    city: str | None = None
    test_type: str | None = None


class ReportSearchMatch(BaseModel):
    """A semantically matched report, joined back to its full record."""

    report: ReportRead
    score: float = Field(description="Cosine similarity of the match.")
    matched_summary: str = Field(
        description=(
            "The sanitised summary that was embedded and matched. Shown so a "
            "user can see why a report was returned."
        )
    )


class ReportSearchResponse(BaseModel):
    query: str
    results: list[ReportSearchMatch] = Field(default_factory=list)
    retrieval_count: int = 0


class ReportIndexResponse(BaseModel):
    """Result of rebuilding the semantic report index."""

    indexed: int
    collection: str
