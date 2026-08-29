"""HTTP middleware: correlation ids and structured access logging."""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from .logging import get_logger
from .request_context import REQUEST_ID_HEADER, set_request_id, set_user_id

logger = get_logger("app.access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign/propagate ``X-Request-ID`` and emit one access log per request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        set_request_id(request_id)
        set_user_id(None)
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # The exception handlers own the response body; we only time it.
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

        # Health checks are noisy; log them at debug level only.
        log = logger.debug if request.url.path in {"/health", "/"} else logger.info
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
