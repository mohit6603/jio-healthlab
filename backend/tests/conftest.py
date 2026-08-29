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
from app.models import RefreshToken, Report, User  # noqa: F401 - register tables
from app.security import Role
from app.services import auth_service


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


#: Password used for every fixture account.
TEST_PASSWORD = "TestPassword!2026"


@pytest.fixture(name="raw_client")
def raw_client_fixture(engine) -> Generator[TestClient, None, None]:
    """Unauthenticated client. Use this to test auth itself."""
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


def _make_user(db: Session, role: Role) -> User:
    return auth_service.create_user(
        db,
        email=f"{role.value.lower()}@test.example.com",
        full_name=f"{role.value.title()} User",
        password=TEST_PASSWORD,
        role=role,
    )


@pytest.fixture(name="user_factory")
def user_factory_fixture(engine):
    """Create a user of a given role and return an authenticated client."""
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def _factory(client: TestClient, role: Role) -> TestClient:
        session = session_factory()
        try:
            existing = auth_service.get_user_by_email(
                session, f"{role.value.lower()}@test.example.com"
            )
            if existing is None:
                _make_user(session, role)
        finally:
            session.close()

        response = client.post(
            "/api/auth/login",
            json={
                "email": f"{role.value.lower()}@test.example.com",
                "password": TEST_PASSWORD,
            },
        )
        assert response.status_code == 200, response.text
        token = response.json()["access_token"]
        client.headers["Authorization"] = f"Bearer {token}"
        return client

    return _factory


@pytest.fixture(name="client")
def client_fixture(raw_client, user_factory) -> TestClient:
    """Client authenticated as ADMIN.

    Most tests are about behaviour other than authorisation, so the default
    client has every permission. RBAC itself is covered explicitly in
    ``test_rbac.py``.
    """
    return user_factory(raw_client, Role.ADMIN)


@pytest.fixture(name="as_role")
def as_role_fixture(raw_client, user_factory):
    """Authenticate the shared client as a specific role."""

    def _login(role: Role) -> TestClient:
        raw_client.headers.pop("Authorization", None)
        return user_factory(raw_client, role)

    return _login


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
