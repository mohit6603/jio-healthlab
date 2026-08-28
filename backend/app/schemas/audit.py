"""Audit trail schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AuditLogRead(BaseModel):
    """One audited action."""

    id: int
    user_id: int | None = None
    user_role: str | None = None
    action: str
    status: str
    resource_type: str | None = None
    resource_id: str | None = None
    detail: str | None = Field(
        default=None,
        description="Short, non-sensitive context. Never a patient identifier "
        "or a prompt.",
    )
    request_id: str | None = None
    latency_ms: float | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AIQueryLogRead(BaseModel):
    """Metadata about one AI interaction.

    Contains no question, answer or retrieved text by design.
    """

    id: int
    user_id: int | None = None
    query_type: str
    success: bool
    error_code: str | None = None
    latency_ms: float | None = None
    source_count: int | None = None
    grounded: bool | None = None
    model: str | None = None
    request_id: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AIUsageByType(BaseModel):
    query_type: str
    count: int
    failures: int
    average_latency_ms: float


class AIUsageSummary(BaseModel):
    """Aggregate AI usage, for the admin view."""

    total: int
    by_type: list[AIUsageByType] = Field(default_factory=list)
