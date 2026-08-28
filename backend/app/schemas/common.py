"""Shared response schemas."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness payload returned by ``GET /health``."""

    status: str = Field(description="``ok`` when the service is healthy.")
    service: str = Field(description="Logical service name (``backend``).")
    app: str = Field(description="Human-readable application name.")
    environment: str = Field(description="Deployment environment.")
    version: str = Field(description="Application version.")


class MessageResponse(BaseModel):
    """Generic acknowledgement payload."""

    message: str
