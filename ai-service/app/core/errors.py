"""Standardised error contract for the AI service.

Mirrors the backend envelope so the proxy layer can forward codes verbatim::

    {"error": {"code": "MODEL_UNAVAILABLE", "message": "..."}}
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
HTTP_422_UNPROCESSABLE: int = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable explanation.")
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """The single error envelope used by every non-2xx response."""

    error: ErrorDetail


class AIError(Exception):
    """Base class for deliberate AI-service failures."""

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


class ModelUnavailableError(AIError):
    """Raised when an embedding or generation model cannot be loaded."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "MODEL_UNAVAILABLE"
    message = (
        "The requested model is not available. Retrieval-only features remain usable."
    )


class GenerationDisabledError(AIError):
    """Raised when generation is switched off via ``LLM_PROVIDER=none``."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "GENERATION_DISABLED"
    message = "Answer generation is disabled in this deployment."


class VectorStoreError(AIError):
    """Raised when Qdrant is unreachable or rejects an operation."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "VECTOR_STORE_UNAVAILABLE"
    message = "The vector database is unavailable."


class DocumentError(AIError):
    """Raised for unreadable, oversized or unsupported documents."""

    status_code = status.HTTP_400_BAD_REQUEST
    code = "INVALID_DOCUMENT"
    message = "The document could not be processed."


class DocumentNotFoundError(AIError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "DOCUMENT_NOT_FOUND"
    message = "No document with that identifier is indexed."


class ModelNotTrainedError(AIError):
    """Raised when an ML artifact has not been produced yet."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "ML_MODEL_NOT_TRAINED"
    message = "No trained model artifact is available. Run `python -m app.ml.train`."


_STATUS_CODES: dict[int, str] = {
    400: "BAD_REQUEST",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
    500: "INTERNAL_ERROR",
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

    @app.exception_handler(AIError)
    async def _ai_error(_: Request, exc: AIError) -> JSONResponse:
        logger.warning(
            "ai_error", extra={"code": exc.code, "status": exc.status_code}
        )
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


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Malformed request or document"},
    404: {"model": ErrorResponse, "description": "Resource not found"},
    422: {"model": ErrorResponse, "description": "Validation failed"},
    503: {"model": ErrorResponse, "description": "Model or vector store unavailable"},
}
