"""Request-scoped context.

A single correlation id (``request_id``) is attached to every inbound request
and propagated to downstream services (notably the AI service) so a failure can
be traced across microservice boundaries.
"""

from contextvars import ContextVar

REQUEST_ID_HEADER = "X-Request-ID"

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_user_id: ContextVar[int | None] = ContextVar("user_id", default=None)


def set_request_id(request_id: str | None) -> None:
    _request_id.set(request_id)


def get_request_id() -> str | None:
    return _request_id.get()


def set_user_id(user_id: int | None) -> None:
    _user_id.set(user_id)


def get_user_id() -> int | None:
    return _user_id.get()
