"""Authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from ..config import Settings, get_settings
from ..core.errors import ERROR_RESPONSES, AppError
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
from ..services import audit_service, auth_service
from ..services.audit_service import Action, Status

router = APIRouter(prefix="/api/auth", tags=["Authentication"], responses=ERROR_RESPONSES)


def _set_refresh_cookie(response: Response, token: str, settings: Settings) -> None:
    """Store the refresh token in an httpOnly cookie.

    httpOnly means JavaScript cannot read it, so an XSS that runs in the page
    cannot exfiltrate a session. SameSite=Strict means a cross-site request
    cannot carry it, which is what a CSRF attack would need. The path scopes
    it to the auth endpoints, so it is not attached to every API call.
    """
    if not settings.refresh_cookie_enabled:
        return

    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.refresh_cookie_samesite,  # type: ignore[arg-type]
        path=settings.refresh_cookie_path,
        max_age=settings.refresh_token_expire_days * 24 * 60 * 60,
    )


def _clear_refresh_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        key=settings.refresh_cookie_name,
        path=settings.refresh_cookie_path,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.refresh_cookie_samesite,  # type: ignore[arg-type]
    )


def _read_refresh_token(
    request: Request, supplied: str | None, settings: Settings
) -> str | None:
    """Prefer the cookie; fall back to the body.

    The body is still accepted so a non-browser client -- a CLI, a test, a
    mobile app that cannot hold cookies -- keeps working.
    """
    if settings.refresh_cookie_enabled:
        cookie = request.cookies.get(settings.refresh_cookie_name)
        if cookie:
            return cookie
    return supplied


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
def login(payload: LoginRequest, db: DbSession, response: Response) -> TokenResponse:
    settings = get_settings()
    try:
        user = auth_service.authenticate(db, payload.email, payload.password, settings)
    except AppError as exc:
        # The attempted email is not recorded: it is unverified user input and
        # would put an address in the audit trail for anyone who typed one.
        audit_service.record(
            db,
            action=Action.LOGIN_FAILED,
            status=Status.FAILURE,
            detail=exc.code,
        )
        raise

    session = auth_service.issue_session(db, user)
    _set_refresh_cookie(response, str(session["refresh_token"]), settings)
    audit_service.record(
        db, action=Action.LOGIN_SUCCEEDED, user=user, resource_type="user",
        resource_id=user.id,
    )
    return TokenResponse(**session)


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
def refresh(
    request: Request,
    response: Response,
    db: DbSession,
    payload: RefreshRequest | None = None,
) -> TokenResponse:
    settings = get_settings()
    token = _read_refresh_token(
        request, payload.refresh_token if payload else None, settings
    )
    if not token:
        raise auth_service.InvalidRefreshTokenError(
            "No refresh token was supplied."
        )

    try:
        session = auth_service.rotate_session(db, token)
    except AppError as exc:
        audit_service.record(
            db,
            action=(
                Action.SESSION_REUSE_DETECTED
                if "security" in exc.message.lower()
                else Action.SESSION_REFRESHED
            ),
            status=Status.FAILURE,
            detail=exc.code,
        )
        raise

    _set_refresh_cookie(response, str(session["refresh_token"]), settings)
    audit_service.record(
        db, action=Action.SESSION_REFRESHED, user=session["user"],
    )
    return TokenResponse(**session)


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
    request: Request,
    response: Response,
    payload: LogoutRequest,
    user: CurrentUser,
    db: DbSession,
) -> LogoutResponse:
    settings = get_settings()
    token = _read_refresh_token(request, payload.refresh_token, settings)

    if payload.all_sessions or token is None:
        ended = auth_service.revoke_all_for_user(db, user.id)
    else:
        ended = 1 if auth_service.revoke_session(db, token) else 0

    # Always clear the cookie, even if the token was already revoked: leaving
    # a dead cookie in the browser only causes a confusing failure later.
    _clear_refresh_cookie(response, settings)

    audit_service.record(
        db, action=Action.LOGOUT, user=user, detail=f"sessions_ended={ended}"
    )
    return LogoutResponse(sessions_ended=ended)


@router.post(
    "/users",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Requires(Permission.ADMIN_USERS)],
    summary="Create a user",
    description="Administrators only. The password is stored as an argon2id "
    "hash and is never returned.",
)
def create_user(
    payload: UserCreate, db: DbSession, actor: CurrentUser
) -> UserRead:
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
    audit_service.record(
        db,
        action=Action.USER_CREATED,
        user=actor,
        resource_type="user",
        resource_id=user.id,
        detail=f"role={user.role}",
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
