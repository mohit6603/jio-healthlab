"""Local generation via Hugging Face Transformers.

Built for CPU inference, which shapes three decisions:

* **Lazy loading.** Weights are a multi-gigabyte download; the container must
  become healthy without them and only pay the cost on first generation.
* **A wall-clock budget.** A 1.5B model on CPU produces a handful of tokens per
  second. Without a time limit a single request can hold a worker for minutes,
  so generation stops at ``LLM_TIMEOUT_SECONDS`` and reports
  ``finish_reason="timeout"`` rather than hanging.
* **Explicit prompt budgeting.** The prompt is truncated to leave room for the
  answer, and the truncation is reported instead of silently discarding
  retrieved context.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..config import Settings, get_settings
from ..core import runtime
from ..core.errors import ModelUnavailableError
from ..core.logging import get_logger
from ..utils.optional import require
from .provider import GenerationResult, LLMProvider

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = get_logger(__name__)


class LocalTransformerProvider(LLMProvider):
    """Runs a causal language model in-process with Transformers."""

    name = "local"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: Any = None
        self._tokenizer: Any = None

    # ---------------------------------------------------------- metadata ---
    @property
    def model_name(self) -> str:
        return self._settings.llm_model

    @property
    def is_available(self) -> bool:
        from ..utils.optional import is_installed

        return is_installed("transformers") and is_installed("torch")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ----------------------------------------------------------- loading ---
    def _load(self) -> tuple[Any, Any]:
        """Load tokenizer and weights once, recording the outcome."""
        if self._model is not None and self._tokenizer is not None:
            # Keep the registry consistent with reality: it is a separate
            # store, and /health must not report an unloaded model that is in
            # fact resident.
            if not runtime.is_loaded(runtime.GENERATION):
                runtime.mark_loaded(runtime.GENERATION)
            return self._tokenizer, self._model

        require("transformers", "Text generation")
        require("torch", "Text generation")

        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self._settings.llm_model)
            model = AutoModelForCausalLM.from_pretrained(
                self._settings.llm_model,
                dtype=torch.float32 if self._settings.llm_device == "cpu" else "auto",
            )
            model.to(self._settings.llm_device)
            model.eval()
        # Any failure here (no network, no disk, OOM) must degrade the service
        # rather than crash it -- retrieval has to keep working.
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            runtime.mark_failed(runtime.GENERATION, reason)
            logger.error(
                "generation_model_load_failed",
                extra={"model": self._settings.llm_model},
            )
            raise ModelUnavailableError(
                f"Could not load generation model '{self._settings.llm_model}'. "
                "Retrieval-only features remain available.",
                details={"reason": reason},
            ) from exc

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._tokenizer = tokenizer
        self._model = model
        runtime.mark_loaded(runtime.GENERATION)
        logger.info(
            "generation_model_loaded",
            extra={
                "model": self._settings.llm_model,
                "device": self._settings.llm_device,
                "load_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return tokenizer, model

    def warm_up(self) -> None:
        self._load()

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        runtime.mark_unloaded(runtime.GENERATION)

    # -------------------------------------------------------- generation ---
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        import torch

        tokenizer, model = self._load()
        settings = self._settings
        limit = max_new_tokens or settings.llm_max_new_tokens
        temp = settings.llm_temperature if temperature is None else temperature

        text = self._render_chat(tokenizer, prompt, system)
        inputs, truncated = self._encode(tokenizer, text, limit)

        started = time.perf_counter()
        stopper = _TimeBudget(settings.llm_timeout_seconds)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": limit,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "stopping_criteria": stopper.as_list(),
        }
        # Greedy decoding when temperature is 0: sampling params are invalid
        # then, and grounded answers benefit from determinism anyway.
        if temp and temp > 0:
            generate_kwargs.update(
                do_sample=True, temperature=temp, top_p=settings.llm_top_p
            )
        else:
            generate_kwargs["do_sample"] = False

        try:
            with torch.inference_mode():
                output = model.generate(**inputs, **generate_kwargs)
        # Generation can fail at runtime (OOM, numerical issues); the caller
        # must get a typed error, not a raw torch exception.
        except Exception as exc:
            logger.error(
                "generation_failed", extra={"model": settings.llm_model}
            )
            raise ModelUnavailableError(
                "Text generation failed.",
                code="GENERATION_FAILED",
                details={"reason": f"{type(exc).__name__}: {exc}"},
            ) from exc

        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        generated = output[0][prompt_tokens:]
        completion_tokens = int(generated.shape[-1])
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

        if stopper.expired:
            finish_reason = "timeout"
        elif completion_tokens >= limit:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        logger.info(
            "generation_completed",
            extra={
                "model": settings.llm_model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
                "finish_reason": finish_reason,
                "prompt_truncated": truncated,
            },
        )

        return GenerationResult(
            text=answer,
            provider=self.name,
            model=settings.llm_model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            prompt_truncated=truncated,
        )

    # ------------------------------------------------------------ helpers ---
    def _render_chat(self, tokenizer: Any, prompt: str, system: str | None) -> str:
        """Apply the model's chat template when it has one."""
        if getattr(tokenizer, "chat_template", None):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # No template: fall back to a plain instruction layout.
        return f"{system}\n\n{prompt}\n\nAnswer:" if system else f"{prompt}\n\nAnswer:"

    def _encode(
        self, tokenizer: Any, text: str, max_new_tokens: int
    ) -> tuple[dict[str, Any], bool]:
        """Tokenise, leaving room in the context window for the answer."""
        budget = max(self._settings.llm_max_input_tokens - max_new_tokens, 128)

        encoded = tokenizer(text, return_tensors="pt")
        length = int(encoded["input_ids"].shape[-1])
        if length <= budget:
            return dict(encoded), False

        # Keep the tail: the question and the generation prompt live at the end.
        logger.warning(
            "prompt_truncated",
            extra={"prompt_tokens": length, "budget": budget},
        )
        truncated = {
            key: value[:, -budget:]
            for key, value in encoded.items()
            if hasattr(value, "shape")
        }
        return truncated, True


class _TimeBudget:
    """Stopping criterion that ends generation after a wall-clock budget.

    CPU generation is slow enough that an unbounded request can occupy a worker
    for minutes. Implemented directly rather than with a version-specific
    built-in so it works across Transformers releases.
    """

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._started = time.perf_counter()
        self.expired = False

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        if time.perf_counter() - self._started > self._seconds:
            self.expired = True
            return True
        return False

    def as_list(self) -> Any:
        from transformers import StoppingCriteriaList

        return StoppingCriteriaList([self])
