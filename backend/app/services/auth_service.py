"""Authentication: sign-in, session rotation, sign-out.

Session design
--------------
A short-lived access JWT plus a long-lived, revocable refresh token. The
refresh token is opaque random bytes; only its SHA-256 digest is stored, so a
database leak alone cannot be replayed as a session.

Refresh rotates: every use issues a new token and revokes the old one. If a
revoked token is presented again -- the signature of a stolen token being
replayed -- every session for that user is revoked, because we cannot tell
whether the attacker or the user holds the current one.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..core.errors import AuthenticationError
from ..core.logging import get_logger
from ..models import RefreshToken, User
from ..security import (
    Role,
    create_access_token,
    hash_password,
    needs_rehash,
    verify_password,
)

logger = get_logger(__name__)

#: Bytes of entropy in a refresh token.
_TOKEN_BYTES = 48


class InvalidCredentialsError(AuthenticationError):
    code = "INVALID_CREDENTIALS"
    #: Deliberately identical for unknown email and wrong password, so the
    #: endpoint cannot be used to enumerate accounts.
    message = "Incorrect email or password."


class InactiveAccountError(AuthenticationError):
    code = "ACCOUNT_INACTIVE"
    message = "This account has been deactivated."


class InvalidRefreshTokenError(AuthenticationError):
    code = "INVALID_REFRESH_TOKEN"
    message = "Your session is no longer valid. Sign in again."


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def hash_token(token: str) -> str:
    """SHA-256 of a refresh token. Fast by design: this is a random 384-bit
    secret, not a password, so it needs no key-stretching."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ users ---
def get_user_by_email(db: Session, email: str) -> User | None:
    return db.execute(
        select(User).where(User.email == email.strip().lower())
    ).scalar_one_or_none()


def create_user(
    db: Session,
    *,
    email: str,
    full_name: str,
    password: str,
    role: Role | str,
    is_active: bool = True,
) -> User:
    """Create a user with a hashed password."""
    user = User(
        email=email.strip().lower(),
        full_name=full_name.strip(),
        password_hash=hash_password(password),
        role=str(Role(role)),
        is_active=is_active,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("user_created", extra={"user_id": user.id, "role": user.role})
    return user


# ---------------------------------------------------------- authentication --
def authenticate(db: Session, email: str, password: str) -> User:
    """Verify credentials and return the user, or raise."""
    user = get_user_by_email(db, email)

    if user is None:
        # Hash anyway so the response time does not reveal whether the account
        # exists.
        hash_password(password)
        logger.info("login_failed", extra={"reason": "unknown_email"})
        raise InvalidCredentialsError()

    if not verify_password(password, user.password_hash):
        logger.info("login_failed", extra={"user_id": user.id, "reason": "bad_password"})
        raise InvalidCredentialsError()

    if not user.is_active:
        logger.info("login_failed", extra={"user_id": user.id, "reason": "inactive"})
        raise InactiveAccountError()

    # Transparently upgrade the hash if the cost parameters have moved on.
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
        logger.info("password_rehashed", extra={"user_id": user.id})

    user.last_login_at = _now()
    db.commit()
    logger.info("login_succeeded", extra={"user_id": user.id, "role": user.role})
    return user


# ------------------------------------------------------------- sessions -----
def issue_refresh_token(
    db: Session, user: User, settings: Settings | None = None
) -> tuple[str, RefreshToken]:
    """Mint a refresh token, storing only its digest."""
    settings = settings or get_settings()
    token = secrets.token_urlsafe(_TOKEN_BYTES)

    record = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(token),
        expires_at=_now() + timedelta(days=settings.refresh_token_expire_days),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return token, record


def issue_session(
    db: Session, user: User, settings: Settings | None = None
) -> dict[str, object]:
    """Issue an access token and a refresh token for ``user``."""
    settings = settings or get_settings()
    access_token, expires_at = create_access_token(
        user_id=user.id, role=user.role, settings=settings
    )
    refresh_token, _ = issue_refresh_token(db, user, settings)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.access_token_expire_minutes * 60,
        "expires_at": expires_at,
        "user": user,
    }


def _find_refresh_record(db: Session, token: str) -> RefreshToken | None:
    return db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == hash_token(token))
    ).scalar_one_or_none()


def revoke_all_for_user(db: Session, user_id: int) -> int:
    """Revoke every live session for a user. Returns how many."""
    records = list(
        db.execute(
            select(RefreshToken).where(
                RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None)
            )
        ).scalars()
    )
    now = _now()
    for record in records:
        record.revoked_at = now
    db.commit()
    return len(records)


def rotate_session(
    db: Session, refresh_token: str, settings: Settings | None = None
) -> dict[str, object]:
    """Exchange a refresh token for a new session, revoking the old token."""
    settings = settings or get_settings()
    record = _find_refresh_record(db, refresh_token)

    if record is None:
        logger.info("refresh_failed", extra={"reason": "unknown_token"})
        raise InvalidRefreshTokenError()

    if record.revoked_at is not None:
        # A revoked token was presented again. Either it was replayed after
        # rotation, or it was stolen. We cannot tell which party holds the
        # current token, so end every session for this user.
        revoked = revoke_all_for_user(db, record.user_id)
        logger.warning(
            "refresh_token_reuse_detected",
            extra={"user_id": record.user_id, "sessions_revoked": revoked},
        )
        raise InvalidRefreshTokenError(
            "This session was already used and has been ended for your "
            "security. Sign in again."
        )

    if record.expires_at <= _now():
        logger.info("refresh_failed", extra={"reason": "expired"})
        raise InvalidRefreshTokenError()

    user = db.get(User, record.user_id)
    if user is None or not user.is_active:
        raise InactiveAccountError()

    new_token, new_record = issue_refresh_token(db, user, settings)
    record.revoked_at = _now()
    record.replaced_by_id = new_record.id
    db.commit()

    access_token, expires_at = create_access_token(
        user_id=user.id, role=user.role, settings=settings
    )
    logger.info("session_rotated", extra={"user_id": user.id})

    return {
        "access_token": access_token,
        "refresh_token": new_token,
        "token_type": "bearer",
        "expires_in": settings.access_token_expire_minutes * 60,
        "expires_at": expires_at,
        "user": user,
    }


def revoke_session(db: Session, refresh_token: str) -> bool:
    """Sign out one session. Unknown tokens are not an error."""
    record = _find_refresh_record(db, refresh_token)
    if record is None or record.revoked_at is not None:
        return False

    record.revoked_at = _now()
    db.commit()
    logger.info("session_revoked", extra={"user_id": record.user_id})
    return True
