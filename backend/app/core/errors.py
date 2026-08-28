"""Standardised API error contract.

Every error the API returns has the same shape::

    {"error": {"code": "REPORT_NOT_FOUND", "message": "...", "details": {...}}}

Tracebacks are never sent to clients. In production the message for an
unhandled exception is deliberately generic; the detail goes to the log stream
keyed by ``request_id``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .logging import get_logger
from .request_context import REQUEST_ID_HEADER, get_request_id

logger = get_logger(__name__)

#: Starlette renamed 422 in newer releases; support both spellings.
HTTP_422_UNPROCESSABLE: int = getattr(
    status, "HTTP_422_UNPROCESSABLE_CONTENT", 422
)


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable explanation.")
    details: dict[str, Any] | None = Field(
        default=None, description="Optional structured context."
    )


class ErrorResponse(BaseModel):
    """The single error envelope used by every non-2xx response."""

    error: ErrorDetail


class AppError(Exception):
    """Base class for errors that map onto a deliberate HTTP response."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "INTERNAL_ERROR"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.status_code = status_code or self.status_code
        self.details = details
        super().__init__(self.message)


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "NOT_FOUND"
    message = "The requested resource was not found."


class ValidationError(AppError):
    status_code = HTTP_422_UNPROCESSABLE
    code = "VALIDATION_ERROR"
    message = "The request payload is invalid."


class AuthenticationError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "NOT_AUTHENTICATED"
    message = "Authentication credentials are missing or invalid."


class PermissionDeniedError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "PERMISSION_DENIED"
    message = "You do not have permission to perform this action."


class AIServiceUnavailableError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "AI_SERVICE_UNAVAILABLE"
    message = "The AI service is temporarily unavailable."


class AIServiceError(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY
    code = "AI_SERVICE_ERROR"
    message = "The AI service returned an unexpected response."


# HTTP status -> default error code, for plain HTTPExceptions raised by FastAPI.
_STATUS_CODES: dict[int, str] = {
    400: "BAD_REQUEST",
    401: "NOT_AUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    502: "BAD_GATEWAY",
    503: "SERVICE_UNAVAILABLE",
    504: "GATEWAY_TIMEOUT",
}


def error_payload(
    code: str, message: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message}
    if details:
        body["details"] = details
    return {"error": body}


def _json_error(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    headers = {}
    request_id = get_request_id()
    if request_id:
        headers[REQUEST_ID_HEADER] = request_id
    return JSONResponse(
        status_code=status_code,
        content=error_payload(code, message, details),
        headers=headers,
    )


def register_exception_handlers(app: FastAPI, *, production: bool = False) -> None:
    """Attach the handlers that guarantee the error envelope."""

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("app_error", extra={"code": exc.code, "status": exc.status_code})
        return _json_error(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "HTTP_ERROR")
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return _json_error(exc.status_code, code, detail)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [
            {
                "field": ".".join(str(part) for part in err.get("loc", ())[1:]),
                "message": err.get("msg", "invalid value"),
            }
            for err in exc.errors()
        ]
        return _json_error(
            HTTP_422_UNPROCESSABLE,
            "VALIDATION_ERROR",
            "The request payload is invalid.",
            {"fields": fields},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unhandled_exception",
            extra={"route": request.url.path, "method": request.method},
        )
        message = (
            "An unexpected error occurred."
            if production
            else f"{type(exc).__name__}: {exc}"
        )
        return _json_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", message
        )


# Reusable OpenAPI response blocks so Swagger documents the error contract.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Malformed request"},
    401: {"model": ErrorResponse, "description": "Not authenticated"},
    403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    404: {"model": ErrorResponse, "description": "Resource not found"},
    422: {"model": ErrorResponse, "description": "Validation failed"},
    503: {"model": ErrorResponse, "description": "Dependency unavailable"},
}

NOT_FOUND_RESPONSE: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Resource not found"},
}
