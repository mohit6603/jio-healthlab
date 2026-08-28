"""Health and model-introspection endpoint tests."""

from app.core import runtime


def _component(body: dict, name: str) -> dict:
    return next(item for item in body["components"] if item["name"] == name)


def test_health_returns_service_identity(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "ai-service"
    assert body["status"] in {"ok", "degraded"}
    assert body["version"]


def test_health_lists_model_components(client):
    body = client.get("/health").json()

    names = {item["name"] for item in body["components"]}
    assert names == {"embedding_model", "generation_model"}


def test_health_reports_missing_optional_dependency_as_unavailable(client):
    """torch/transformers are absent in the lite test environment."""
    body = client.get("/health").json()

    embedding = _component(body, "embedding_model")
    assert embedding["state"] in {"ok", "unavailable"}
    if embedding["state"] == "unavailable":
        assert "sentence_transformers" in embedding["detail"]
        assert body["status"] == "degraded"


def test_health_marks_generation_disabled_when_provider_is_none(settings, client_factory):
    settings.llm_provider = "none"
    client = client_factory(settings)

    body = client.get("/health").json()
    generation = _component(body, "generation_model")

    assert generation["state"] == "disabled"
    assert "none" in generation["detail"]


def test_health_surfaces_recorded_load_failure(client):
    runtime.mark_failed(runtime.EMBEDDING, "out of memory")

    body = client.get("/health").json()

    assert body["status"] == "degraded"


def test_health_echoes_request_id(client):
    response = client.get("/health", headers={"X-Request-ID": "trace-xyz"})

    assert response.headers["X-Request-ID"] == "trace-xyz"


def test_health_generates_request_id_when_absent(client):
    assert client.get("/health").headers.get("X-Request-ID")


def test_models_lists_all_three_roles(client):
    body = client.get("/models").json()

    roles = {item["role"] for item in body["models"]}
    assert roles == {"embedding", "generation", "ml"}


def test_models_reports_configured_names(client, settings):
    body = client.get("/models").json()
    by_role = {item["role"]: item for item in body["models"]}

    assert by_role["embedding"]["name"] == settings.embedding_model
    assert by_role["embedding"]["dimension"] == settings.embedding_dimension
    assert by_role["generation"]["name"] == settings.llm_model
    assert by_role["ml"]["name"] == settings.delay_model_name


def test_models_reports_nothing_loaded_at_rest(client):
    body = client.get("/models").json()

    assert all(item["loaded"] is False for item in body["models"])


def test_models_reflects_runtime_load_state(client):
    runtime.mark_loaded(runtime.EMBEDDING)

    body = client.get("/models").json()
    embedding = next(item for item in body["models"] if item["role"] == "embedding")

    assert embedding["loaded"] is True


def test_models_marks_generation_unavailable_when_disabled(settings, client_factory):
    settings.llm_provider = "none"
    client = client_factory(settings)

    body = client.get("/models").json()
    generation = next(item for item in body["models"] if item["role"] == "generation")

    assert generation["available"] is False
    assert "disabled" in generation["detail"].lower()


def test_endpoints_do_not_load_weights(client):
    """Neither endpoint may flip a model into the loaded state."""
    client.get("/health")
    client.get("/models")

    assert not runtime.is_loaded(runtime.EMBEDDING)
    assert not runtime.is_loaded(runtime.GENERATION)
    assert not runtime.is_loaded(runtime.DELAY_MODEL)
