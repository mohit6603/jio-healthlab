"""JIO HealthLab backend -- application factory and wiring.

Routing, business logic and persistence are deliberately kept in separate
packages (``routers`` / ``services`` / ``models``). This module only assembles
them.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, get_settings
from .core.errors import register_exception_handlers
from .core.logging import configure_logging, get_logger
from .core.middleware import RequestContextMiddleware
from .routers import ai, audit, auth, dashboard, reports, system
from .routers.system import APP_VERSION
from .services.ai_client import close_ai_client

DESCRIPTION = """
Diagnostics report operations API for **JIO HealthLab**.

* **Reports** -- create, search, update and track diagnostic orders.
* **Dashboard** -- aggregated operational metrics.
* **Authentication** -- sign in, rotate a session, inspect permissions.
* **System** -- liveness probe and service metadata.
* **AI** -- knowledge assistant and semantic search.

> AI output is informational only. It is **not** a medical diagnosis.
> An AI outage never affects reports or the dashboard.

Every error response uses the envelope
`{"error": {"code": "...", "message": "..."}}`.

Except for `/health`, `/` and `/api/auth/login`, every endpoint requires
a bearer access token from `POST /api/auth/login`. Authorisation is by
permission, granted through the caller's role.
"""

OPENAPI_TAGS = [
    {"name": "System", "description": "Health checks and service metadata."},
    {
        "name": "Authentication",
        "description": (
            "Sign-in, session rotation and user administration. All other "
            "endpoints require a bearer access token."
        ),
    },
    {"name": "Reports", "description": "Laboratory report lifecycle."},
    {"name": "Dashboard", "description": "Aggregated operational metrics."},
    {
        "name": "AI",
        "description": (
            "Knowledge assistant and semantic search, proxied to the "
            "internal AI service. AI output is informational only and is "
            "not a medical diagnosis."
        ),
    },
]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI application."""
    settings = settings or get_settings()
    # Fail fast rather than run production with a placeholder signing key.
    settings.assert_production_ready()

    configure_logging(level=settings.log_level, json_output=settings.log_json)
    logger = get_logger(__name__)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info(
            "application_started",
            extra={"environment": settings.environment, "version": APP_VERSION},
        )
        yield
        # Release the pooled AI-service connections on shutdown.
        await close_ai_client()
        logger.info("application_stopping")

    app = FastAPI(
        title=settings.app_name,
        version=APP_VERSION,
        description=DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
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
    app.include_router(auth.router)
    app.include_router(reports.router)
    app.include_router(dashboard.router)
    app.include_router(ai.router)
    app.include_router(audit.router)

    return app


app = create_app()
