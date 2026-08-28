"""Health and model-introspection endpoints.

Neither endpoint loads model weights -- they answer from configuration, the
runtime registry and cheap import probes, so orchestrators can poll them
freely.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..config import Settings
from ..core import runtime
from ..schemas.common import (
    ComponentHealth,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
)
from ..utils.optional import is_installed
from .deps import SettingsDep

router = APIRouter(tags=["System"])

APP_VERSION = "1.0.0"

#: Import name -> human label, for dependency probes.
_EMBEDDING_MODULE = "sentence_transformers"
_GENERATION_MODULE = "transformers"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Returns 200 whenever the process can serve traffic. `status` is "
        "`degraded` when an optional component (typically the generation "
        "model) is unavailable but retrieval still works."
    ),
)
def health(settings: SettingsDep) -> HealthResponse:
    components = _component_health(settings)
    degraded = any(item.state in {"degraded", "unavailable"} for item in components)

    return HealthResponse(
        status="degraded" if degraded else "ok",
        service="ai-service",
        app=settings.app_name,
        environment=settings.environment,
        version=APP_VERSION,
        components=components,
    )


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="Configured models",
    description=(
        "Reports which embedding, generation and ML models this deployment is "
        "configured to use, and whether their weights are currently resident."
    ),
)
def models(settings: SettingsDep) -> ModelsResponse:
    return ModelsResponse(models=_model_info(settings))


def _component_health(settings: Settings) -> list[ComponentHealth]:
    """Describe every optional component the service depends on."""
    components: list[ComponentHealth] = []

    # --- embeddings: required for every retrieval path ---
    embedding_failure = runtime.failure_reason(runtime.EMBEDDING)
    if not is_installed(_EMBEDDING_MODULE):
        components.append(
            ComponentHealth(
                name="embedding_model",
                state="unavailable",
                detail=f"'{_EMBEDDING_MODULE}' is not installed in this image.",
            )
        )
    elif embedding_failure:
        components.append(
            ComponentHealth(
                name="embedding_model", state="unavailable", detail=embedding_failure
            )
        )
    else:
        components.append(ComponentHealth(name="embedding_model", state="ok"))

    # --- generation: optional, retrieval must survive without it ---
    if not settings.generation_enabled:
        components.append(
            ComponentHealth(
                name="generation_model",
                state="disabled",
                detail="LLM_PROVIDER is set to 'none'.",
            )
        )
    elif not is_installed(_GENERATION_MODULE):
        components.append(
            ComponentHealth(
                name="generation_model",
                state="unavailable",
                detail=f"'{_GENERATION_MODULE}' is not installed in this image.",
            )
        )
    elif (failure := runtime.failure_reason(runtime.GENERATION)) is not None:
        components.append(
            ComponentHealth(
                name="generation_model", state="unavailable", detail=failure
            )
        )
    else:
        components.append(ComponentHealth(name="generation_model", state="ok"))

    return components


def _model_info(settings: Settings) -> list[ModelInfo]:
    embedding_available = is_installed(_EMBEDDING_MODULE)
    generation_installed = is_installed(_GENERATION_MODULE)

    return [
        ModelInfo(
            role="embedding",
            name=settings.embedding_model,
            provider="sentence-transformers",
            loaded=runtime.is_loaded(runtime.EMBEDDING),
            available=embedding_available,
            device=settings.embedding_device,
            dimension=settings.embedding_dimension,
            detail=runtime.failure_reason(runtime.EMBEDDING)
            or (
                None
                if embedding_available
                else f"'{_EMBEDDING_MODULE}' is not installed."
            ),
        ),
        ModelInfo(
            role="generation",
            name=settings.llm_model,
            provider=settings.llm_provider,
            loaded=runtime.is_loaded(runtime.GENERATION),
            available=settings.generation_enabled and generation_installed,
            device=settings.llm_device,
            detail=_generation_detail(settings, generation_installed),
        ),
        ModelInfo(
            role="ml",
            name=settings.delay_model_name,
            provider="scikit-learn",
            loaded=runtime.is_loaded(runtime.DELAY_MODEL),
            available=is_installed("sklearn"),
            detail=runtime.failure_reason(runtime.DELAY_MODEL),
        ),
    ]


def _generation_detail(settings: Settings, installed: bool) -> str | None:
    if not settings.generation_enabled:
        return "Generation is disabled (LLM_PROVIDER=none)."
    if not installed:
        return f"'{_GENERATION_MODULE}' is not installed."
    return runtime.failure_reason(runtime.GENERATION)
