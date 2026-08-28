"""Shared response schemas for the AI service."""

from typing import Literal

from pydantic import BaseModel, Field

ComponentState = Literal["ok", "degraded", "unavailable", "disabled"]


class ComponentHealth(BaseModel):
    """Health of one dependency the AI service relies on."""

    name: str = Field(description="Component identifier, e.g. ``qdrant``.")
    state: ComponentState
    detail: str | None = Field(
        default=None, description="Why the component is not ``ok``."
    )


class HealthResponse(BaseModel):
    """Liveness payload returned by ``GET /health``.

    ``status`` stays ``ok`` while the process can serve requests even if an
    optional component (the generation model, say) is unavailable -- retrieval
    must keep working. It flips to ``degraded`` so operators can see the
    difference.
    """

    status: Literal["ok", "degraded"]
    service: str = "ai-service"
    app: str
    environment: str
    version: str
    components: list[ComponentHealth] = Field(default_factory=list)


class ModelInfo(BaseModel):
    """Description of a configured model."""

    role: Literal["embedding", "generation", "ml"]
    name: str = Field(description="Model identifier or artifact name.")
    provider: str = Field(description="Where the model comes from.")
    loaded: bool = Field(description="Whether weights are resident in memory.")
    available: bool = Field(
        description="Whether the model can be loaded in this deployment."
    )
    device: str | None = None
    dimension: int | None = Field(
        default=None, description="Embedding width, for embedding models."
    )
    detail: str | None = None


class ModelsResponse(BaseModel):
    """Payload returned by ``GET /models``."""

    models: list[ModelInfo]


class ReadinessResponse(BaseModel):
    """Payload returned by ``GET /health/ready``.

    Separate from ``/health`` on purpose: liveness must never fail because a
    downstream dependency is down, or an orchestrator would restart a process
    that is working perfectly well. Readiness is where dependency probes live.
    """

    status: Literal["ready", "degraded", "not_ready"]
    service: str = "ai-service"
    components: list[ComponentHealth] = Field(default_factory=list)
