"""Authentication schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from ..security import MIN_PASSWORD_LENGTH, Role


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1, max_length=512)


class LogoutRequest(BaseModel):
    refresh_token: str | None = Field(
        default=None,
        description="Session to end. Omit to end every session for this user.",
    )
    all_sessions: bool = Field(
        default=False, description="End every session for this user."
    )


class UserRead(BaseModel):
    """A user as returned by the API. Never includes the password hash."""

    id: int
    email: EmailStr
    full_name: str
    role: Role
    is_active: bool
    last_login_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=120)
    password: str = Field(
        min_length=MIN_PASSWORD_LENGTH,
        max_length=256,
        description=f"At least {MIN_PASSWORD_LENGTH} characters.",
    )
    role: Role = Role.VIEWER


class TokenResponse(BaseModel):
    """Issued session.

    The refresh token is returned once, here. It is stored server-side only as
    a SHA-256 digest and cannot be retrieved again.
    """

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Access-token lifetime in seconds.")
    expires_at: datetime
    user: UserRead


class MeResponse(BaseModel):
    """The signed-in user and what they may do."""

    user: UserRead
    permissions: list[str] = Field(
        description="Permission strings granted by this user's role."
    )


class LogoutResponse(BaseModel):
    sessions_ended: int
