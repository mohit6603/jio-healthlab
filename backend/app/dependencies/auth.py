"""Authentication and authorisation dependencies.

Routes declare the permission they need::

    @router.post("/reports", dependencies=[Requires(Permission.REPORTS_CREATE)])

The role that happens to hold that permission is decided in
``security/permissions.py`` and nowhere else.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..core.errors import AuthenticationError, PermissionDeniedError
from ..core.logging import get_logger
from ..core.request_context import set_user_id
from ..models import User
from ..security import Permission, decode_access_token, has_permission
from .db import DbSession

logger = get_logger(__name__)

#: auto_error=False so a missing header produces our error envelope rather
#: than FastAPI's default body.
_bearer = HTTPBearer(auto_error=False, description="Bearer access token.")


async def get_current_user(
    request: Request,
    db: DbSession,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(_bearer)
    ] = None,
) -> User:
    """Resolve the signed-in user from the bearer token."""
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Sign in to access this resource.")

    payload = decode_access_token(credentials.credentials)

    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthenticationError() from exc

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        # A token issued before the account was deactivated must stop working.
        raise AuthenticationError("This account is no longer active.")

    # The role in the token could be stale; the database is authoritative.
    request.state.user = user
    set_user_id(user.id)
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


class RequirePermission:
    """Dependency that enforces a single permission."""

    def __init__(self, permission: Permission) -> None:
        self.permission = permission

    def __call__(self, user: CurrentUser) -> User:
        if not has_permission(user.role, self.permission):
            logger.info(
                "permission_denied",
                extra={
                    "user_id": user.id,
                    "role": user.role,
                    "permission": str(self.permission),
                },
            )
            raise PermissionDeniedError(
                f"Your role ({user.role}) is not permitted to perform this "
                "action."
            )
        return user


# Capitalised deliberately: it reads as a declaration at the route.
def Requires(permission: Permission):
    """Route dependency: ``dependencies=[Requires(Permission.REPORTS_READ)]``."""
    return Depends(RequirePermission(permission))
