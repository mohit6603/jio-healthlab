"""Authentication tests: login, session rotation, logout, token handling."""

from datetime import timedelta

import pytest

from app.security import Role, create_access_token
from app.security.passwords import hash_password, needs_rehash, verify_password
from tests.conftest import TEST_PASSWORD

ADMIN_EMAIL = "admin@test.example.com"


@pytest.fixture(name="admin")
def admin_fixture(raw_client, user_factory):
    user_factory(raw_client, Role.ADMIN)
    raw_client.headers.pop("Authorization", None)
    return raw_client


def login(client, email=ADMIN_EMAIL, password=TEST_PASSWORD):
    return client.post("/api/auth/login", json={"email": email, "password": password})


# ------------------------------------------------------------- passwords ---
def test_password_hash_is_not_the_password():
    assert hash_password("SuperSecret!2026") != "SuperSecret!2026"


def test_password_hashes_are_salted():
    assert hash_password("same") != hash_password("same")


def test_correct_password_verifies():
    assert verify_password("SuperSecret!2026", hash_password("SuperSecret!2026"))


def test_wrong_password_does_not_verify():
    assert not verify_password("wrong", hash_password("SuperSecret!2026"))


def test_corrupt_hash_returns_false_rather_than_raising():
    assert verify_password("anything", "not-a-hash") is False


def test_argon2id_is_used():
    assert hash_password("x").startswith("$argon2id$")


def test_current_hash_does_not_need_rehashing():
    assert needs_rehash(hash_password("x")) is False


# ----------------------------------------------------------------- login ---
def test_login_returns_a_session(admin):
    response = login(admin)

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0
    assert body["user"]["email"] == ADMIN_EMAIL
    assert body["user"]["role"] == "ADMIN"


def test_login_never_returns_the_password_hash(admin):
    assert "password" not in login(admin).text.lower().replace("password!", "")


def test_wrong_password_is_rejected(admin):
    response = login(admin, password="not-the-password")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_CREDENTIALS"


def test_unknown_email_gives_the_same_error_as_a_wrong_password(admin):
    """Login must not reveal which accounts exist."""
    unknown = login(admin, email="nobody@test.example.com")
    wrong = login(admin, password="not-the-password")

    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json()


def test_email_is_case_insensitive(admin):
    assert login(admin, email=ADMIN_EMAIL.upper()).status_code == 200


def test_inactive_account_cannot_sign_in(admin, db):
    from app.services import auth_service

    user = auth_service.get_user_by_email(db, ADMIN_EMAIL)
    user.is_active = False
    db.commit()

    response = login(admin)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "ACCOUNT_INACTIVE"


def test_login_records_the_time(admin, db):
    from app.services import auth_service

    login(admin)

    assert auth_service.get_user_by_email(db, ADMIN_EMAIL).last_login_at is not None


def test_malformed_login_payload_is_rejected(admin):
    assert admin.post("/api/auth/login", json={"email": "not-an-email"}).status_code == 422


# ------------------------------------------------------------ protection ---
def test_protected_route_requires_a_token(admin):
    response = admin.get("/api/reports")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "NOT_AUTHENTICATED"


def test_health_stays_public(admin):
    assert admin.get("/health").status_code == 200


def test_login_stays_public(admin):
    assert login(admin).status_code == 200


def test_valid_token_grants_access(admin):
    token = login(admin).json()["access_token"]

    response = admin.get("/api/reports", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200


def test_garbage_token_is_rejected(admin):
    response = admin.get(
        "/api/reports", headers={"Authorization": "Bearer not.a.real.token"}
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_TOKEN"


def test_expired_token_is_rejected(admin, db):
    from app.services import auth_service

    user = auth_service.get_user_by_email(db, ADMIN_EMAIL)
    token, _ = create_access_token(
        user_id=user.id, role=user.role, expires_in=timedelta(seconds=-1)
    )

    response = admin.get("/api/reports", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
    assert "expired" in response.json()["error"]["message"].lower()


def test_token_signed_with_another_key_is_rejected(admin, db):
    import jwt

    from app.services import auth_service

    user = auth_service.get_user_by_email(db, ADMIN_EMAIL)
    forged = jwt.encode(
        {"sub": str(user.id), "role": "ADMIN", "type": "access", "exp": 9999999999,
         "iat": 1, "iss": "jio-healthlab"},
        "attacker-key-long-enough-for-hs256-abcdefgh",
        algorithm="HS256",
    )

    response = admin.get("/api/reports", headers={"Authorization": f"Bearer {forged}"})

    assert response.status_code == 401


def test_refresh_token_is_not_accepted_as_an_access_token(admin):
    refresh_token = login(admin).json()["refresh_token"]

    response = admin.get(
        "/api/reports", headers={"Authorization": f"Bearer {refresh_token}"}
    )

    assert response.status_code == 401


def test_deactivating_a_user_invalidates_their_existing_token(admin, db):
    from app.services import auth_service

    token = login(admin).json()["access_token"]
    user = auth_service.get_user_by_email(db, ADMIN_EMAIL)
    user.is_active = False
    db.commit()

    response = admin.get("/api/reports", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
    assert "no longer active" in response.json()["error"]["message"]


# --------------------------------------------------------------- refresh ---
def test_refresh_issues_a_new_session(admin):
    refresh_token = login(admin).json()["refresh_token"]

    response = admin.post("/api/auth/refresh", json={"refresh_token": refresh_token})

    assert response.status_code == 200
    assert response.json()["access_token"]


def test_refresh_rotates_the_token(admin):
    first = login(admin).json()["refresh_token"]

    second = admin.post(
        "/api/auth/refresh", json={"refresh_token": first}
    ).json()["refresh_token"]

    assert second != first


def test_the_old_refresh_token_stops_working(admin):
    first = login(admin).json()["refresh_token"]
    admin.post("/api/auth/refresh", json={"refresh_token": first})

    response = admin.post("/api/auth/refresh", json={"refresh_token": first})

    assert response.status_code == 401


def test_reusing_a_rotated_token_ends_every_session(admin):
    """Replay is indistinguishable from theft, so all sessions are revoked."""
    first = login(admin).json()["refresh_token"]
    second = admin.post(
        "/api/auth/refresh", json={"refresh_token": first}
    ).json()["refresh_token"]

    reuse = admin.post("/api/auth/refresh", json={"refresh_token": first})
    after = admin.post("/api/auth/refresh", json={"refresh_token": second})

    assert reuse.status_code == 401
    assert "security" in reuse.json()["error"]["message"].lower()
    # Even the legitimate current token is now dead.
    assert after.status_code == 401


def test_unknown_refresh_token_is_rejected(admin):
    response = admin.post("/api/auth/refresh", json={"refresh_token": "made-up"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_REFRESH_TOKEN"


def test_refresh_tokens_are_stored_hashed(admin, db):
    from app.models import RefreshToken

    refresh_token = login(admin).json()["refresh_token"]

    stored = db.query(RefreshToken).all()
    assert stored
    assert all(record.token_hash != refresh_token for record in stored)
    assert all(len(record.token_hash) == 64 for record in stored)


# ------------------------------------------------------------------- me ----
def test_me_returns_the_user_and_permissions(admin):
    token = login(admin).json()["access_token"]

    body = admin.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {token}"}
    ).json()

    assert body["user"]["email"] == ADMIN_EMAIL
    assert "reports:read" in body["permissions"]
    assert "admin:users" in body["permissions"]


def test_me_requires_a_token(admin):
    assert admin.get("/api/auth/me").status_code == 401


# ---------------------------------------------------------------- logout ---
def test_logout_revokes_the_session(admin):
    session = login(admin).json()
    headers = {"Authorization": f"Bearer {session['access_token']}"}

    response = admin.post(
        "/api/auth/logout",
        json={"refresh_token": session["refresh_token"]},
        headers=headers,
    )
    reuse = admin.post(
        "/api/auth/refresh", json={"refresh_token": session["refresh_token"]}
    )

    assert response.status_code == 200
    assert response.json()["sessions_ended"] == 1
    assert reuse.status_code == 401


def test_logout_all_sessions(admin):
    first = login(admin).json()
    second = login(admin).json()

    response = admin.post(
        "/api/auth/logout",
        json={"all_sessions": True},
        headers={"Authorization": f"Bearer {second['access_token']}"},
    )

    # At least the two opened here; the fixture's own login counts too.
    assert response.json()["sessions_ended"] >= 2
    for session in (first, second):
        assert admin.post(
            "/api/auth/refresh", json={"refresh_token": session["refresh_token"]}
        ).status_code == 401


def test_logout_with_an_unknown_token_is_not_an_error(admin):
    token = login(admin).json()["access_token"]

    response = admin.post(
        "/api/auth/logout",
        json={"refresh_token": "made-up"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["sessions_ended"] == 0


def test_logout_requires_authentication(admin):
    assert admin.post("/api/auth/logout", json={}).status_code == 401
