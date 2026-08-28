"""Provider selection.

``LLM_PROVIDER`` picks the backend. Adding a hosted provider means writing an
:class:`LLMProvider` subclass and registering it here -- nothing in the RAG
pipeline, prompts or retrieval changes.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import Settings, get_settings
from ..core.errors import AIError
from ..core.logging import get_logger
from .local_transformer import LocalTransformerProvider
from .provider import DisabledProvider, LLMProvider

logger = get_logger(__name__)

#: Aliases that all mean "generation is switched off".
DISABLED_ALIASES = frozenset({"none", "off", "disabled", ""})

ProviderFactory = Callable[[Settings], LLMProvider]

#: Registry of available backends. Hosted providers (OpenAI, Bedrock) slot in
#: here once implemented; the rest of the codebase is unaffected.
_REGISTRY: dict[str, ProviderFactory] = {
    "local": lambda settings: LocalTransformerProvider(settings),
    "transformers": lambda settings: LocalTransformerProvider(settings),
}


def available_providers() -> list[str]:
    return sorted({*_REGISTRY, *DISABLED_ALIASES} - {""})


def create_provider(settings: Settings | None = None) -> LLMProvider:
    """Build the provider named by ``LLM_PROVIDER``."""
    settings = settings or get_settings()
    key = settings.llm_provider.strip().lower()

    if key in DISABLED_ALIASES:
        logger.info("generation_disabled", extra={"reason": "LLM_PROVIDER=none"})
        return DisabledProvider()

    factory = _REGISTRY.get(key)
    if factory is None:
        raise AIError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
            f"Available: {', '.join(available_providers())}.",
            code="UNKNOWN_LLM_PROVIDER",
            status_code=500,
            details={"available": available_providers()},
        )

    provider = factory(settings)
    logger.info(
        "llm_provider_selected",
        extra={"provider": provider.name, "model": provider.model_name},
    )
    return provider


_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    """Process-wide singleton -- weights are loaded at most once."""
    global _provider
    if _provider is None:
        _provider = create_provider()
    return _provider


def reset_llm_provider() -> None:
    """Drop the singleton (test helper)."""
    global _provider
    if _provider is not None:
        _provider.close()
    _provider = None
