"""Shared fixtures for AI-service tests.

Unit tests run without torch / transformers / sentence-transformers installed,
which keeps CI fast and proves the graceful-degradation paths actually work.
"""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.core import runtime
from app.llm import create_provider, get_llm_provider
from app.main import create_app


@pytest.fixture(autouse=True)
def clean_runtime() -> Generator[None, None, None]:
    """Model load-state must not leak between tests."""
    runtime.reset()
    yield
    runtime.reset()


@pytest.fixture(name="settings")
def settings_fixture() -> Settings:
    return Settings(
        environment="test",
        log_json=False,
        qdrant_url="http://qdrant.invalid:6333",
        qdrant_collection="test_knowledge",
        qdrant_report_collection="test_reports",
    )


@pytest.fixture(name="client")
def client_fixture(settings: Settings) -> Generator[TestClient, None, None]:
    app = create_app(settings)
    _wire(app, settings)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _wire(app, custom: Settings) -> None:
    """Point every settings-derived dependency at the test settings.

    ``get_settings`` is cached and the LLM provider is a module singleton built
    from the global config. Without this, a test app configured with different
    settings would be internally inconsistent -- /models would report the
    process-wide provider rather than the one this app is configured for.
    """
    app.dependency_overrides[get_settings] = lambda: custom
    app.dependency_overrides[get_llm_provider] = lambda: create_provider(custom)


@pytest.fixture(name="client_factory")
def client_factory_fixture():
    """Build a client for a modified settings object mid-test."""
    created: list[TestClient] = []

    def _factory(custom: Settings) -> TestClient:
        app = create_app(custom)
        _wire(app, custom)
        client = TestClient(app)
        client.__enter__()
        created.append(client)
        return client

    yield _factory

    for client in created:
        client.__exit__(None, None, None)
