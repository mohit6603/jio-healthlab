"""Generation tests against real model weights.

Skipped unless transformers is installed AND the generation model is already
present in the Hugging Face cache -- these tests must never trigger a
multi-gigabyte download as a side effect of running the suite::

    pytest -m integration tests/test_llm_integration.py
"""

from __future__ import annotations

import os

import pytest

from app.config import Settings
from app.core import runtime
from app.llm.local_transformer import LocalTransformerProvider
from app.utils.optional import is_installed

#: Small enough to keep these tests to a few seconds on CPU.
TEST_MODEL = os.getenv("LLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (is_installed("transformers") and is_installed("torch")),
        reason="transformers/torch are not installed",
    ),
]


@pytest.fixture(name="provider", scope="module")
def provider_fixture():
    provider = LocalTransformerProvider(
        Settings(
            llm_provider="local",
            llm_model=TEST_MODEL,
            llm_max_new_tokens=48,
            llm_temperature=0.0,
            llm_timeout_seconds=180,
            log_json=False,
        )
    )
    provider.warm_up()
    yield provider
    provider.close()


SYSTEM = (
    "Answer using ONLY the reference material. If it does not contain the "
    "answer, say you do not have enough information."
)


# -------------------------------------------------------------- mechanics --
def test_provider_loads_and_reports_state(provider):
    # warm_up is idempotent and re-syncs the runtime registry, which the
    # autouse reset fixture clears between tests.
    provider.warm_up()

    assert provider.is_loaded is True
    assert provider.is_available is True
    assert runtime.is_loaded(runtime.GENERATION) is True


def test_generate_produces_text(provider):
    result = provider.generate("Say the single word: hello", system=SYSTEM)

    assert result.text.strip()
    assert result.provider == "local"
    assert result.model == TEST_MODEL


def test_generate_reports_token_counts(provider):
    result = provider.generate("Count to three.", system=SYSTEM)

    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.latency_ms > 0


def test_max_new_tokens_is_respected(provider):
    result = provider.generate("Write a long essay about blood.", max_new_tokens=8)

    assert result.completion_tokens <= 8
    assert result.finish_reason in {"length", "stop", "timeout"}


def test_greedy_decoding_is_deterministic(provider):
    first = provider.generate("Name one blood cell type.", temperature=0.0)
    second = provider.generate("Name one blood cell type.", temperature=0.0)

    assert first.text == second.text


def test_answer_excludes_the_prompt(provider):
    """Only newly generated tokens are decoded, never the prompt echo."""
    marker = "PROMPT_MARKER_XYZZY"
    result = provider.generate(f"{marker}. Reply with the word acknowledged.")

    assert marker not in result.text


def test_model_is_loaded_once(provider):
    before = provider._model

    provider.generate("anything")

    assert provider._model is before


# --------------------------------------------------------------- budgets --
def test_time_budget_stops_a_long_generation():
    """A tiny wall-clock budget must cut generation short, not hang."""
    provider = LocalTransformerProvider(
        Settings(
            llm_model=TEST_MODEL,
            llm_max_new_tokens=512,
            llm_timeout_seconds=1.0,
            llm_temperature=0.0,
            log_json=False,
        )
    )

    result = provider.generate("Write an extremely long essay about haematology.")

    assert result.finish_reason == "timeout"
    assert result.latency_ms < 60_000


def test_oversized_prompt_is_truncated_not_rejected():
    provider = LocalTransformerProvider(
        Settings(
            llm_model=TEST_MODEL,
            llm_max_new_tokens=8,
            llm_max_input_tokens=256,
            llm_timeout_seconds=120,
            llm_temperature=0.0,
            log_json=False,
        )
    )
    huge = " ".join(f"Filler sentence number {i}." for i in range(500))

    result = provider.generate(f"{huge}\n\nQuestion: what is this?")

    assert result.prompt_truncated is True
    assert result.prompt_tokens <= 256


# ------------------------------------------------------------ grounding ---
def test_model_uses_supplied_context(provider):
    prompt = (
        "Reference material:\n"
        "The Zenithal Marker Index (ZMI) is measured in blue units.\n\n"
        "Question: What units is the Zenithal Marker Index measured in?"
    )

    result = provider.generate(prompt, system=SYSTEM)

    assert "blue" in result.text.lower()
