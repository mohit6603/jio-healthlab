"""Request-scoped correlation context.

The backend forwards its ``X-Request-ID`` so a single user action can be traced
across both services in the log stream.
"""

from contextvars import ContextVar

REQUEST_ID_HEADER = "X-Request-ID"

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str | None) -> None:
    _request_id.set(request_id)


def get_request_id() -> str | None:
    return _request_id.get()
