"""JIO HealthLab backend -- application factory and wiring.

Routing, business logic and persistence are deliberately kept in separate
packages (``routers`` / ``services`` / ``models``). This module only assembles
them.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, get_settings
from .core.errors import register_exception_handlers
from .core.logging import configure_logging, get_logger
from .core.middleware import RequestContextMiddleware
from .routers import dashboard, reports, system
from .routers.system import APP_VERSION

DESCRIPTION = """
Diagnostics report operations API for **JIO HealthLab**.

* **Reports** -- create, search, update and track diagnostic orders.
* **Dashboard** -- aggregated operational metrics.
* **System** -- liveness probe and service metadata.

Every error response uses the envelope
`{"error": {"code": "...", "message": "..."}}`.
"""

OPENAPI_TAGS = [
    {"name": "System", "description": "Health checks and service metadata."},
    {"name": "Reports", "description": "Laboratory report lifecycle."},
    {"name": "Dashboard", "description": "Aggregated operational metrics."},
]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI application."""
    settings = settings or get_settings()

    configure_logging(level=settings.log_level, json_output=settings.log_json)
    logger = get_logger(__name__)

    app = FastAPI(
        title=settings.app_name,
        version=APP_VERSION,
        description=DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app, production=settings.is_production)

    app.include_router(system.router)
    app.include_router(reports.router)
    app.include_router(dashboard.router)

    logger.info(
        "application_started",
        extra={"environment": settings.environment, "version": APP_VERSION},
    )
    return app


app = create_app()
