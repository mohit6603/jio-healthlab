"""HTTP client for the AI microservice.

The reports system must not depend on the AI service being up. Every failure
mode here is translated into a typed, user-presentable error:

============================  ===============================================
httpx.ConnectError/Timeout    503 AI_SERVICE_UNAVAILABLE
httpx.ReadTimeout             504 AI_SERVICE_TIMEOUT
AI service 4xx/5xx            the AI service's own error code, forwarded
malformed body                502 AI_SERVICE_ERROR
============================  ===============================================

Forwarding the AI service's code matters: the browser needs to distinguish
"generation is switched off" from "the vector store is down", and those
distinctions are made upstream.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..config import Settings, get_settings
from ..core.errors import AIServiceError, AIServiceUnavailableError, AppError
from ..core.logging import get_logger
from ..core.request_context import REQUEST_ID_HEADER, get_request_id

logger = get_logger(__name__)


class AIServiceTimeoutError(AppError):
    """The AI service accepted the request but did not answer in time."""

    status_code = 504
    code = "AI_SERVICE_TIMEOUT"
    message = (
        "The AI service took too long to respond. Generation on CPU can be "
        "slow; try a shorter question or retry."
    )


class AIServiceClient:
    """Async client for the AI service.

    One :class:`httpx.AsyncClient` is shared for the process lifetime so
    connections are pooled rather than re-established per request.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._owns_client = client is None

    @property
    def base_url(self) -> str:
        return self._settings.ai_service_url.rstrip("/")

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    self._settings.ai_service_timeout_seconds,
                    connect=self._settings.ai_service_connect_timeout_seconds,
                ),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    # ------------------------------------------------------------ requests --
    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Call the AI service and normalise every failure mode."""
        headers = {}
        # Propagate the correlation id so one user action can be traced across
        # both services in the log stream.
        if request_id := get_request_id():
            headers[REQUEST_ID_HEADER] = request_id

        try:
            response = await self._get_client().request(
                method,
                path,
                json=json,
                headers=headers,
                timeout=timeout,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            logger.warning(
                "ai_service_unreachable", extra={"path": path, "url": self.base_url}
            )
            raise AIServiceUnavailableError(
                "The AI assistant is currently unavailable. Reports and the "
                "dashboard are unaffected."
            ) from exc
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
            logger.warning("ai_service_timeout", extra={"path": path})
            raise AIServiceTimeoutError() from exc
        except httpx.HTTPError as exc:
            logger.warning("ai_service_transport_error", extra={"path": path})
            raise AIServiceError(
                "The AI service could not be reached.",
                details={"reason": type(exc).__name__},
            ) from exc

        if response.status_code >= 400:
            raise self._translate_error(response)

        try:
            payload = response.json()
        except ValueError as exc:
            logger.error("ai_service_bad_json", extra={"path": path})
            raise AIServiceError(
                "The AI service returned a malformed response."
            ) from exc

        if not isinstance(payload, dict):
            raise AIServiceError("The AI service returned an unexpected payload.")
        return payload

    def _translate_error(self, response: httpx.Response) -> AppError:
        """Forward the AI service's error code rather than flattening it."""
        code = "AI_SERVICE_ERROR"
        message = "The AI service returned an error."

        try:
            body = response.json()
            error = body.get("error") if isinstance(body, dict) else None
            if isinstance(error, dict):
                code = str(error.get("code") or code)
                message = str(error.get("message") or message)
        except ValueError:
            pass

        logger.warning(
            "ai_service_error",
            extra={"status": response.status_code, "code": code},
        )

        # 5xx from the AI service is a dependency failure for us; 4xx is a
        # problem with the request we forwarded, so the status is preserved.
        status_code = response.status_code if response.status_code < 500 else (
            response.status_code if response.status_code in {503, 504} else 502
        )
        return AppError(message, code=code, status_code=status_code)

    # ------------------------------------------------------------- methods --
    async def health(self) -> dict[str, Any]:
        """Probe the AI service. Never raises -- returns a report."""
        try:
            payload = await self._request(
                "GET", "/health", timeout=self._settings.ai_health_timeout_seconds
            )
        except AppError as exc:
            return {
                "reachable": False,
                "status": "unreachable",
                "detail": exc.message,
                "components": [],
            }

        return {
            "reachable": True,
            "status": payload.get("status", "ok"),
            "detail": None,
            "components": payload.get("components", []),
        }

    async def chat(
        self,
        question: str,
        *,
        top_k: int | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Ask a grounded question."""
        return await self._request(
            "POST",
            "/rag/query",
            json=_compact(
                {"question": question, "top_k": top_k, "category": category}
            ),
        )

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Semantic search over the knowledge base."""
        return await self._request(
            "POST",
            "/rag/search",
            json=_compact({"query": query, "top_k": top_k, "category": category}),
        )

    async def predict_delay_batch(
        self, items: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Score many requests in one call.

        The dashboard needs a risk score per visible report; one round trip per
        report would make the page unusable.
        """
        return await self._request(
            "POST", "/ml/predict-delay/batch", json={"items": items}
        )

    async def explain_report(
        self,
        report_summary: str,
        *,
        search_text: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Explain a laboratory request.

        ``report_summary`` must already have passed through
        :func:`app.services.sanitizer.sanitize_report`. This method does not
        sanitise -- it is the transport, not the boundary.
        """
        return await self._request(
            "POST",
            "/rag/explain",
            json=_compact(
                {
                    "report_summary": report_summary,
                    "search_text": search_text,
                    "top_k": top_k,
                }
            ),
        )


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop unset optional fields so the AI service applies its own defaults."""
    return {key: value for key, value in payload.items() if value is not None}


_client: AIServiceClient | None = None


def get_ai_client() -> AIServiceClient:
    """Process-wide singleton, so the connection pool is reused."""
    global _client
    if _client is None:
        _client = AIServiceClient()
    return _client


async def close_ai_client() -> None:
    """Release the shared client (called on application shutdown)."""
    global _client
    if _client is not None:
        await _client.aclose()
    _client = None
