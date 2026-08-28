"""LLM abstraction tests.

The provider interface, the factory and the disabled path are covered without
transformers installed. Real generation is exercised in
``test_llm_integration.py``.
"""

from __future__ import annotations

import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from app.config import Settings
from app.core import runtime
from app.core.errors import AIError, GenerationDisabledError, ModelUnavailableError
from app.llm import (
    DisabledProvider,
    GenerationResult,
    LLMProvider,
    available_providers,
    create_provider,
    get_llm_provider,
)
from app.llm.factory import reset_llm_provider
from app.llm.local_transformer import LocalTransformerProvider, _TimeBudget


# ------------------------------------------------------------- interface ---
def test_disabled_provider_is_an_llm_provider():
    assert isinstance(DisabledProvider(), LLMProvider)


def test_disabled_provider_reports_unavailable():
    provider = DisabledProvider()

    assert provider.is_available is False
    assert provider.is_loaded is False
    assert provider.model_name == "disabled"


def test_disabled_provider_raises_on_generate():
    with pytest.raises(GenerationDisabledError) as excinfo:
        DisabledProvider().generate("anything")

    assert excinfo.value.code == "GENERATION_DISABLED"
    assert excinfo.value.status_code == 503


def test_generation_result_defaults():
    result = GenerationResult(text="hi", provider="local", model="m")

    assert result.finish_reason == "stop"
    assert result.prompt_truncated is False
    assert result.completion_tokens == 0


# --------------------------------------------------------------- factory ---
@pytest.mark.parametrize("value", ["none", "None", "OFF", "disabled", ""])
def test_disabled_aliases_yield_the_disabled_provider(value):
    provider = create_provider(Settings(llm_provider=value, log_json=False))

    assert isinstance(provider, DisabledProvider)


@pytest.mark.parametrize("value", ["local", "LOCAL", " transformers "])
def test_local_aliases_yield_the_transformer_provider(value):
    provider = create_provider(Settings(llm_provider=value, log_json=False))

    assert isinstance(provider, LocalTransformerProvider)


def test_unknown_provider_is_rejected_with_the_available_list():
    with pytest.raises(AIError) as excinfo:
        create_provider(Settings(llm_provider="gpt-9000", log_json=False))

    assert excinfo.value.code == "UNKNOWN_LLM_PROVIDER"
    assert "local" in excinfo.value.details["available"]


def test_available_providers_lists_local_and_none():
    providers = available_providers()

    assert "local" in providers
    assert "none" in providers


def test_factory_returns_a_singleton():
    reset_llm_provider()
    try:
        assert get_llm_provider() is get_llm_provider()
    finally:
        reset_llm_provider()


def test_model_selection_comes_from_settings():
    provider = create_provider(
        Settings(llm_provider="local", llm_model="some/other-model", log_json=False)
    )

    assert provider.model_name == "some/other-model"


# ----------------------------------------------------------- time budget ---
def test_time_budget_allows_generation_within_the_limit():
    budget = _TimeBudget(seconds=10)

    assert budget(None, None) is False
    assert budget.expired is False


def test_time_budget_stops_once_exceeded():
    budget = _TimeBudget(seconds=0.01)
    time.sleep(0.02)

    assert budget(None, None) is True
    assert budget.expired is True


# --------------------------------------------------- local provider unit ---
class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def __getitem__(self, item):
        # (row, col) indexing, as used when truncating the prompt.
        if isinstance(item, tuple):
            return FakeTensor(self.values[item[1]])
        if isinstance(item, slice):
            return FakeTensor(self.values[item])
        return FakeTensor(self.values)


class FakeTokenizer:
    chat_template = "template"
    pad_token_id = 0
    eos_token_id = 2
    pad_token = None
    eos_token = "<eos>"

    def __init__(self, state):
        self._state = state
        self.rendered: list[list[dict]] = []
        self.encoded: list[str] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.rendered.append(messages)
        return "RENDERED:" + " | ".join(m["content"] for m in messages)

    def __call__(self, text, return_tensors=None):
        # Prompt length is driven by the fixture so tests can control it.
        self.encoded.append(text)
        return {"input_ids": FakeTensor(list(range(self._state.prompt_len)))}

    def decode(self, tokens, skip_special_tokens=True):
        return "  generated answer  "


@pytest.fixture(name="fake_transformers")
def fake_transformers_fixture(monkeypatch):
    """Install fake transformers/torch modules so _load() can be exercised."""
    state = SimpleNamespace(generate_kwargs=None, prompt_len=6, new_tokens=4)

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            state.generate_kwargs = kwargs
            for criterion in kwargs.get("stopping_criteria", []):
                criterion(None, None)
            total = state.prompt_len + state.new_tokens
            return [FakeTensor(list(range(total)))]

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *a, **k: FakeTokenizer(state)
    )
    transformers.AutoModelForCausalLM = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *a, **k: FakeModel()
    )
    transformers.StoppingCriteriaList = list  # type: ignore[attr-defined]
    transformers.__spec__ = SimpleNamespace(name="transformers")  # type: ignore[attr-defined]

    class _InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    torch = ModuleType("torch")
    torch.float32 = "float32"  # type: ignore[attr-defined]
    torch.inference_mode = _InferenceMode  # type: ignore[attr-defined]
    torch.__spec__ = SimpleNamespace(name="torch")  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)

    from app.utils.optional import is_installed

    is_installed.cache_clear()
    yield state
    is_installed.cache_clear()


@pytest.fixture(name="local_provider")
def local_provider_fixture(fake_transformers):
    return LocalTransformerProvider(
        Settings(llm_provider="local", log_json=False, llm_max_new_tokens=32)
    )


def test_local_provider_is_not_loaded_until_used(local_provider):
    assert local_provider.is_loaded is False


def test_generate_returns_stripped_text(local_provider):
    result = local_provider.generate("What does a CBC measure?")

    assert result.text == "generated answer"
    assert result.provider == "local"


def test_generate_records_token_counts_and_latency(local_provider, fake_transformers):
    result = local_provider.generate("question")

    assert result.prompt_tokens == fake_transformers.prompt_len
    assert result.completion_tokens == fake_transformers.new_tokens
    assert result.latency_ms >= 0


def test_generate_applies_the_chat_template_with_system(local_provider):
    local_provider.generate("the question", system="the system prompt")

    tokenizer = local_provider._tokenizer
    roles = [message["role"] for message in tokenizer.rendered[0]]
    assert roles == ["system", "user"]


def test_generate_omits_system_message_when_absent(local_provider):
    local_provider.generate("the question")

    assert [m["role"] for m in local_provider._tokenizer.rendered[0]] == ["user"]


def test_zero_temperature_uses_greedy_decoding(local_provider, fake_transformers):
    local_provider.generate("question", temperature=0)

    assert fake_transformers.generate_kwargs["do_sample"] is False
    assert "temperature" not in fake_transformers.generate_kwargs


def test_positive_temperature_enables_sampling(local_provider, fake_transformers):
    local_provider.generate("question", temperature=0.7)

    assert fake_transformers.generate_kwargs["do_sample"] is True
    assert fake_transformers.generate_kwargs["temperature"] == 0.7


def test_max_new_tokens_override_is_applied(local_provider, fake_transformers):
    local_provider.generate("question", max_new_tokens=17)

    assert fake_transformers.generate_kwargs["max_new_tokens"] == 17


def test_finish_reason_length_when_budget_is_consumed(
    local_provider, fake_transformers
):
    fake_transformers.new_tokens = 32

    result = local_provider.generate("question", max_new_tokens=32)

    assert result.finish_reason == "length"


def test_model_loads_once_across_calls(local_provider):
    local_provider.generate("one")
    first = local_provider._model
    local_provider.generate("two")

    assert local_provider._model is first


def test_loading_marks_runtime_state(local_provider):
    local_provider.warm_up()

    assert runtime.is_loaded(runtime.GENERATION) is True


def test_close_releases_the_model(local_provider):
    local_provider.warm_up()
    local_provider.close()

    assert local_provider.is_loaded is False
    assert runtime.is_loaded(runtime.GENERATION) is False


# ---------------------------------------------------------------- errors ---
def test_load_failure_raises_model_unavailable(monkeypatch, fake_transformers):
    import transformers

    def explode(*args, **kwargs):
        raise OSError("no route to huggingface.co")

    monkeypatch.setattr(
        transformers, "AutoTokenizer", SimpleNamespace(from_pretrained=explode)
    )
    provider = LocalTransformerProvider(Settings(log_json=False))

    with pytest.raises(ModelUnavailableError) as excinfo:
        provider.warm_up()

    assert excinfo.value.status_code == 503
    assert "no route" in excinfo.value.details["reason"]


def test_load_failure_is_recorded_for_health(monkeypatch, fake_transformers):
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        SimpleNamespace(
            from_pretrained=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("oom"))
        ),
    )

    with pytest.raises(ModelUnavailableError):
        LocalTransformerProvider(Settings(log_json=False)).warm_up()

    assert runtime.is_loaded(runtime.GENERATION) is False
    assert "oom" in runtime.failure_reason(runtime.GENERATION)


def test_missing_transformers_reports_clearly(settings):
    """With transformers absent, the message must name the missing module."""
    from app.utils.optional import is_installed

    if is_installed("transformers"):
        pytest.skip("transformers is installed in this environment")

    with pytest.raises(ModelUnavailableError) as excinfo:
        LocalTransformerProvider(settings).warm_up()

    assert "transformers" in excinfo.value.message
