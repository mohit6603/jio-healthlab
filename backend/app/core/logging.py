"""Structured logging.

Emits one JSON object per log record so CloudWatch / Loki / Docker log drivers
can index the fields directly. Falls back to a human-readable formatter when
``LOG_JSON=false`` (nicer during local development).

Sensitive values are redacted centrally: anything that looks like a token,
password, patient identifier or free-text prompt must not reach the log stream.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any

from .request_context import get_request_id, get_user_id

SERVICE_NAME = "backend"

_RESERVED = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)

# Keys whose values are never safe to log verbatim.
_SENSITIVE_KEYS = frozenset(
    {
        "password", "passwd", "secret", "token", "access_token", "refresh_token",
        "authorization", "api_key", "jwt", "jwt_secret_key", "patient_name",
        "phone", "email", "address", "prompt", "question", "answer",
    }
)

_SENSITIVE_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),          # email addresses
    re.compile(r"(?<!\d)(?:\+91[\s-]?)?\d{10}(?!\d)"),  # bare phone numbers
)

REDACTED = "[redacted]"


def redact(value: Any) -> Any:
    """Recursively strip sensitive values out of a log payload."""
    if isinstance(value, dict):
        return {
            key: (REDACTED if key.lower() in _SENSITIVE_KEYS else redact(val))
            for key, val in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        for pattern in _SENSITIVE_PATTERNS:
            value = pattern.sub(REDACTED, value)
        return value
    return value


class JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON document."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "service": SERVICE_NAME,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = get_request_id()
        if request_id:
            payload["request_id"] = request_id

        user_id = get_user_id()
        if user_id is not None:
            payload["user_id"] = user_id

        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(redact(payload), default=str)


class TextFormatter(logging.Formatter):
    """Readable single-line formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        request_id = get_request_id()
        return f"{base} [request_id={request_id}]" if request_id else base


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Install the root logging handler. Safe to call more than once."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if json_output
        else TextFormatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Uvicorn ships its own handlers; route them through ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
