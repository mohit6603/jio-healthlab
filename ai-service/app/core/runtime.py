"""Runtime state for lazily-loaded models.

Heavy models are loaded once, on first use, and kept for the life of the
process. Loaders record their state here so ``/health`` and ``/models`` can
report what is resident without importing torch or touching the weights.
"""

from __future__ import annotations

import threading

#: Registry keys used across the service.
EMBEDDING = "embedding"
GENERATION = "generation"
DELAY_MODEL = "delay_model"

_lock = threading.Lock()
_loaded: dict[str, bool] = {}
_failures: dict[str, str] = {}


def mark_loaded(key: str) -> None:
    """Record that ``key``'s weights are resident in memory."""
    with _lock:
        _loaded[key] = True
        _failures.pop(key, None)


def mark_failed(key: str, reason: str) -> None:
    """Record why ``key`` could not be loaded, for the health payload."""
    with _lock:
        _loaded[key] = False
        _failures[key] = reason


def mark_unloaded(key: str) -> None:
    """Forget ``key`` -- used when a model is released or reset in tests."""
    with _lock:
        _loaded.pop(key, None)
        _failures.pop(key, None)


def is_loaded(key: str) -> bool:
    return _loaded.get(key, False)


def failure_reason(key: str) -> str | None:
    return _failures.get(key)


def reset() -> None:
    """Clear all recorded state (test helper)."""
    with _lock:
        _loaded.clear()
        _failures.clear()
