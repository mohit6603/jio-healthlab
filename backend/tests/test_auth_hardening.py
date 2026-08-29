"""Tests for the httpOnly cookie session and login lockout."""

import pytest

from app.config import get_settings
from app.services import auth_service
from app.services.audit_service import Action
from tests.conftest import TEST_PASSWORD

EMAIL = "admin@test.example.com"


@pytest.fixture(name="anon")
def anon_fixture(raw_client, user_factory):
    from app.security import Role

    user_factory(raw_client, Role.ADMIN)
    raw_client.headers.pop("Authorization", None)
    raw_client.cookies.clear()
    return raw_client


def login(client, password=TEST_PASSWORD):
    return client.post(
        "/api/auth/login", json={"email": EMAIL, "password": password}
    )


# ─────────────────────────────────────────────── httpOnly cookie ──
def test_login_sets_an_httponly_cookie(anon):
    settings = get_settings()

    response = login(anon)

    assert response.status_code == 200
    header = response.headers.get("set-cookie", "")
    assert settings.refresh_cookie_name in header
    # httpOnly is the whole point: JavaScript, including an XSS payload,
    # cannot read it.
    assert "httponly" in header.lower()


def test_cookie_is_samesite_strict(anon):
    header = login(anon).headers.get("set-cookie", "").lower()

    assert "samesite=strict" in header


def test_cookie_is_scoped_to_the_auth_endpoints(anon):
    header = login(anon).headers.get("set-cookie", "").lower()

    assert "path=/api/auth" in header


def test_cookie_is_not_secure_locally(anon):
    """Marking it Secure over plain http would make the browser discard it."""
    header = login(anon).headers.get("set-cookie", "").lower()

    assert "secure" not in header.replace("httponly", "")


def test_refresh_works_from_the_cookie_alone(anon):
    login(anon)

    # No body at all -- the browser flow.
    response = anon.post("/api/auth/refresh")

    assert response.status_code == 200
    assert response.json()["access_token"]


def test_refresh_rotates_the_cookie(anon):
    settings = get_settings()
    login(anon)
    first = anon.cookies.get(settings.refresh_cookie_name)

    anon.post("/api/auth/refresh")

    assert anon.cookies.get(settings.refresh_cookie_name) != first


def test_cookie_wins_over_a_stale_body_token(anon):
    """A client sending both must not be able to downgrade to an old token."""
    first = login(anon).json()["refresh_token"]
    anon.post("/api/auth/refresh")  # rotates; `first` is now revoked

    response = anon.post("/api/auth/refresh", json={"refresh_token": first})

    assert response.status_code == 200


def test_body_token_still_works_for_non_browser_clients(anon):
    token = login(anon).json()["refresh_token"]
    anon.cookies.clear()

    response = anon.post("/api/auth/refresh", json={"refresh_token": token})

    assert response.status_code == 200


def test_refresh_without_any_token_is_rejected(anon):
    response = anon.post("/api/auth/refresh")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_REFRESH_TOKEN"


def test_logout_clears_the_cookie(anon):
    settings = get_settings()
    session = login(anon).json()

    response = anon.post(
        "/api/auth/logout",
        json={},
        headers={"Authorization": f"Bearer {session['access_token']}"},
    )

    assert response.status_code == 200
    assert not anon.cookies.get(settings.refresh_cookie_name)


# ────────────────────────────────────────────────────── lockout ──
def test_account_locks_after_repeated_failures(anon):
    settings = get_settings()

    for _ in range(settings.login_max_attempts):
        login(anon, password="wrong")

    response = login(anon, password="wrong")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "ACCOUNT_LOCKED"


def test_a_locked_account_rejects_even_the_correct_password(anon):
    settings = get_settings()
    for _ in range(settings.login_max_attempts):
        login(anon, password="wrong")

    response = login(anon)

    assert response.json()["error"]["code"] == "ACCOUNT_LOCKED"


def test_failures_below_the_threshold_do_not_lock(anon):
    settings = get_settings()

    for _ in range(settings.login_max_attempts - 1):
        assert login(anon, password="wrong").json()["error"]["code"] == (
            "INVALID_CREDENTIALS"
        )

    assert login(anon).status_code == 200


def test_a_successful_sign_in_clears_the_counter(anon, db):
    """The threshold is consecutive failures, not lifetime ones."""
    for _ in range(3):
        login(anon, password="wrong")
    login(anon)

    user = auth_service.get_user_by_email(db, EMAIL)
    assert user.failed_login_count == 0
    assert user.locked_until is None


def test_lockout_is_recorded_in_the_audit_trail(anon, db):
    from app.services import audit_service

    settings = get_settings()
    for _ in range(settings.login_max_attempts):
        login(anon, password="wrong")

    failures = audit_service.list_audit_logs(
        db, action=str(Action.LOGIN_FAILED), limit=50
    )
    assert len(failures) >= settings.login_max_attempts


def test_lockout_expires(anon, db):
    from datetime import timedelta

    settings = get_settings()
    for _ in range(settings.login_max_attempts):
        login(anon, password="wrong")
    assert login(anon).json()["error"]["code"] == "ACCOUNT_LOCKED"

    # Wind the clock past the lockout rather than sleeping for 15 minutes.
    user = auth_service.get_user_by_email(db, EMAIL)
    user.locked_until = user.locked_until - timedelta(minutes=30)
    db.commit()

    assert login(anon).status_code == 200


def test_an_administrator_can_unlock(anon, db):
    settings = get_settings()
    for _ in range(settings.login_max_attempts):
        login(anon, password="wrong")

    auth_service.unlock_user(db, auth_service.get_user_by_email(db, EMAIL))

    assert login(anon).status_code == 200


def test_lockout_does_not_reveal_whether_an_account_exists(anon):
    """An unknown email must never produce the lockout message."""
    for _ in range(12):
        anon.post(
            "/api/auth/login",
            json={"email": "nobody@test.example.com", "password": "wrong"},
        )

    body = anon.post(
        "/api/auth/login",
        json={"email": "nobody@test.example.com", "password": "wrong"},
    ).json()

    assert body["error"]["code"] == "INVALID_CREDENTIALS"
