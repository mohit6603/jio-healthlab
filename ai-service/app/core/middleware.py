"""HTTP middleware: correlation ids and structured access logging."""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from .logging import get_logger
from .request_context import REQUEST_ID_HEADER, set_request_id

logger = get_logger("app.access")

_QUIET_ROUTES = frozenset({"/health", "/"})


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Adopt the caller's ``X-Request-ID`` (or mint one) and log each request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        set_request_id(request_id)
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_failed",
                extra={
                    "route": request.url.path,
                    "method": request.method,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            raise

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers[REQUEST_ID_HEADER] = request_id

        log = logger.debug if request.url.path in _QUIET_ROUTES else logger.info
        log(
            "request_completed",
            extra={
                "route": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        return response
