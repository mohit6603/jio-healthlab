"""LLM provider interface.

RAG depends on this interface, never on a concrete model class. Swapping the
local Transformers backend for a hosted API (OpenAI, Bedrock, anything else)
means adding a subclass and a factory entry -- no change to prompt building,
retrieval or the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from ..core.errors import GenerationDisabledError


@dataclass(slots=True)
class GenerationResult:
    """One completion, plus the metadata needed to observe and attribute it."""

    text: str
    provider: str
    model: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: ``stop`` (hit an end token), ``length`` (hit max_new_tokens) or
    #: ``timeout`` (hit the wall-clock budget -- common on CPU).
    finish_reason: str = "stop"
    #: True when the prompt had to be shortened to fit the context window.
    prompt_truncated: bool = False


class LLMProvider(ABC):
    """Anything that can turn a prompt into text."""

    #: Short identifier used in logs and responses, e.g. ``local``.
    name: str = "base"

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the model behind this provider."""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether generation can be attempted at all in this deployment."""

    @property
    def is_loaded(self) -> bool:
        """Whether weights are currently resident in memory."""
        return False

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        """Generate a completion for ``prompt``."""

    def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Yield the completion incrementally.

        The default falls back to generating in full and yielding once, so a
        provider without native streaming still satisfies the interface -- the
        caller sees one large chunk rather than an error.
        """
        yield self.generate(
            prompt,
            system=system,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ).text

    @property
    def supports_streaming(self) -> bool:
        """Whether :meth:`stream` yields incrementally rather than at once."""
        return False

    # Optional hooks: a no-op default is correct for providers with no
    # local state (a hosted API has nothing to warm up or release).
    def warm_up(self) -> None:  # noqa: B027 - deliberately optional
        """Load weights ahead of the first request. Optional."""

    def close(self) -> None:  # noqa: B027 - deliberately optional
        """Release resources. Optional."""


class DisabledProvider(LLMProvider):
    """Stands in when ``LLM_PROVIDER=none``.

    Generation is switched off, but retrieval must keep working -- so the
    service still starts, ``/rag/search`` still serves, and only attempts to
    *generate* fail, with an explicit reason.
    """

    name = "none"

    @property
    def model_name(self) -> str:
        return "disabled"

    @property
    def is_available(self) -> bool:
        return False

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        raise GenerationDisabledError()
