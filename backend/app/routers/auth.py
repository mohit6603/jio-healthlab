"""Authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from ..core.errors import ERROR_RESPONSES
from ..dependencies import CurrentUser, DbSession, Requires
from ..schemas.auth import (
    LoginRequest,
    LogoutRequest,
    LogoutResponse,
    MeResponse,
    RefreshRequest,
    TokenResponse,
    UserCreate,
    UserRead,
)
from ..security import Permission, permissions_for
from ..services import auth_service

router = APIRouter(prefix="/api/auth", tags=["Authentication"], responses=ERROR_RESPONSES)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Sign in",
    description=(
        "Exchanges credentials for a short-lived access token and a "
        "long-lived refresh token.\n\n"
        "An unknown email and a wrong password return the identical error, so "
        "this endpoint cannot be used to discover which accounts exist."
    ),
)
def login(payload: LoginRequest, db: DbSession) -> TokenResponse:
    user = auth_service.authenticate(db, payload.email, payload.password)
    return TokenResponse(**auth_service.issue_session(db, user))


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Exchange a refresh token for a new session",
    description=(
        "Rotates the session: a new refresh token is issued and the presented "
        "one is revoked.\n\n"
        "Presenting an already-revoked token ends **every** session for that "
        "user. That is the signature of a replayed token, and it is not "
        "possible to tell whether the legitimate user or an attacker holds "
        "the current one."
    ),
)
def refresh(payload: RefreshRequest, db: DbSession) -> TokenResponse:
    return TokenResponse(**auth_service.rotate_session(db, payload.refresh_token))


@router.get(
    "/me",
    response_model=MeResponse,
    summary="The signed-in user and their permissions",
    description=(
        "Returns the current user together with the permission strings their "
        "role grants, so a client can hide actions it cannot perform. The "
        "server enforces the same list independently."
    ),
)
def me(user: CurrentUser) -> MeResponse:
    return MeResponse(
        user=UserRead.model_validate(user),
        permissions=sorted(str(item) for item in permissions_for(user.role)),
    )


@router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Sign out",
    description=(
        "Revokes the supplied refresh token, or every session for the user "
        "when `all_sessions` is set. Signing out with an unknown token is not "
        "an error."
    ),
)
def logout(
    payload: LogoutRequest, user: CurrentUser, db: DbSession
) -> LogoutResponse:
    if payload.all_sessions or payload.refresh_token is None:
        return LogoutResponse(
            sessions_ended=auth_service.revoke_all_for_user(db, user.id)
        )

    ended = auth_service.revoke_session(db, payload.refresh_token)
    return LogoutResponse(sessions_ended=1 if ended else 0)


@router.post(
    "/users",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Requires(Permission.ADMIN_USERS)],
    summary="Create a user",
    description="Administrators only. The password is stored as an argon2id "
    "hash and is never returned.",
)
def create_user(payload: UserCreate, db: DbSession) -> UserRead:
    from ..core.errors import AppError

    if auth_service.get_user_by_email(db, payload.email) is not None:
        raise AppError(
            "An account with that email already exists.",
            code="EMAIL_ALREADY_REGISTERED",
            status_code=status.HTTP_409_CONFLICT,
        )

    user = auth_service.create_user(
        db,
        email=payload.email,
        full_name=payload.full_name,
        password=payload.password,
        role=payload.role,
    )
    return UserRead.model_validate(user)


@router.get(
    "/users",
    response_model=list[UserRead],
    dependencies=[Requires(Permission.ADMIN_USERS)],
    summary="List users",
    description="Administrators only.",
)
def list_users(db: DbSession) -> list[UserRead]:
    from sqlalchemy import select

    from ..models import User

    users = db.execute(select(User).order_by(User.id)).scalars()
    return [UserRead.model_validate(user) for user in users]
