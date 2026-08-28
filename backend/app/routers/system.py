"""System endpoints: root banner and liveness probe."""

from fastapi import APIRouter

from ..config import get_settings
from ..schemas.common import HealthResponse

router = APIRouter(tags=["System"])

APP_VERSION = "4.0.0"


@router.get("/", summary="Service banner")
def root() -> dict[str, str]:
    """Human-friendly landing payload for the bare API host."""
    settings = get_settings()
    return {
        "app": settings.app_name,
        "message": "Open /docs for the API or run the React frontend.",
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 while the process is able to serve traffic. "
    "Used by Docker/ALB health checks.",
)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service="backend",
        app=settings.app_name,
        environment=settings.environment,
        version=APP_VERSION,
    )
