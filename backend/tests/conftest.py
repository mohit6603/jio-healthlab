"""Shared pytest fixtures.

Tests run against an in-memory SQLite database so they need neither MySQL nor
Docker. Each test gets a fresh schema, which keeps them order-independent.
"""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import create_app
from app.models import Report  # noqa: F401 - registers the table on Base


@pytest.fixture(name="engine")
def engine_fixture():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(name="db")
def db_fixture(engine) -> Generator[Session, None, None]:
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(name="client")
def client_fixture(engine) -> Generator[TestClient, None, None]:
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def override_get_db() -> Generator[Session, None, None]:
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(name="report_payload")
def report_payload_fixture() -> dict:
    return {
        "patient_name": "Asha Nair",
        "age": 34,
        "test_type": "CBC Panel",
        "city": "Mumbai",
        "lab_branch": "BKC Flagship",
        "status": "registered",
        "priority": "urgent",
    }
