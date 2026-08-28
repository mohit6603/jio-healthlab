"""JIO HealthLab AI service -- application factory.

A deliberately separate FastAPI process. Transformers, torch, sentence-
transformers and scikit-learn live here and never leak into the main backend
image, which stays small and quick to deploy.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api import documents, health, rag
from .api.health import APP_VERSION
from .config import Settings, get_settings
from .core.errors import register_exception_handlers
from .core.logging import configure_logging, get_logger
from .core.middleware import RequestContextMiddleware

DESCRIPTION = """
Retrieval-augmented generation and predictive ML for **JIO HealthLab**.

* **RAG** -- semantic search and grounded answers over the lab knowledge base.
* **Documents** -- ingestion and lifecycle of knowledge-base documents.
* **ML** -- report-delay risk prediction.
* **System** -- health and model introspection.

> AI output is informational only. It is **not** a medical diagnosis.

This service is internal: the React frontend talks to the main backend, which
proxies here. Do not expose it publicly.
"""

OPENAPI_TAGS = [
    {"name": "System", "description": "Health checks and model introspection."},
    {"name": "RAG", "description": "Semantic search and grounded question answering."},
    {"name": "Documents", "description": "Knowledge-base document lifecycle."},
    {"name": "ML", "description": "Predictive models for lab operations."},
]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured AI-service application."""
    settings = settings or get_settings()

    configure_logging(level=settings.log_level, json_output=settings.log_json)
    logger = get_logger(__name__)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info(
            "ai_service_starting",
            extra={
                "environment": settings.environment,
                "version": APP_VERSION,
                "embedding_model": settings.embedding_model,
                "llm_provider": settings.llm_provider,
                "generation_enabled": settings.generation_enabled,
            },
        )
        yield
        logger.info("ai_service_stopping")

    app = FastAPI(
        title=settings.app_name,
        version=APP_VERSION,
        description=DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )

    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app, production=settings.is_production)

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(rag.router)

    return app


app = create_app()
