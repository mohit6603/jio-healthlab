"""Access-token issuing and verification.

Access tokens are short-lived JWTs carrying the user id and role. Sessions are
kept alive by a separate opaque refresh token stored server-side, so a stolen
access token expires quickly and a stolen refresh token can be revoked --
neither is possible with a single long-lived JWT.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from ..config import Settings, get_settings
from ..core.errors import AuthenticationError

TOKEN_TYPE_ACCESS = "access"


class TokenError(AuthenticationError):
    """The token is missing, malformed, expired or not ours."""

    code = "INVALID_TOKEN"
    message = "The access token is invalid or has expired."


def create_access_token(
    *,
    user_id: int,
    role: str,
    settings: Settings | None = None,
    expires_in: timedelta | None = None,
) -> tuple[str, datetime]:
    """Issue an access token. Returns the token and its expiry."""
    settings = settings or get_settings()
    now = datetime.now(UTC)
    expires_at = now + (
        expires_in or timedelta(minutes=settings.access_token_expire_minutes)
    )

    payload: dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "type": TOKEN_TYPE_ACCESS,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        # A unique id per token, so an individual token could be denylisted.
        "jti": uuid.uuid4().hex,
        "iss": settings.jwt_issuer,
    }
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token, expires_at


def decode_access_token(token: str, settings: Settings | None = None) -> dict[str, Any]:
    """Verify and decode an access token, or raise :class:`TokenError`."""
    settings = settings or get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "iat", "sub", "type"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Your session has expired. Sign in again.") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError() from exc

    # A refresh token must never be accepted as an access token.
    if payload.get("type") != TOKEN_TYPE_ACCESS:
        raise TokenError()

    return payload
