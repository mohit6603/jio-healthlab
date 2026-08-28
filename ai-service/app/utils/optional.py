"""Optional heavyweight dependency probing.

``torch`` / ``transformers`` / ``sentence-transformers`` are large and are only
present in the AI service image. This module answers "is it installed?" without
importing it, so ``/health`` and ``/models`` stay fast and never trigger a
multi-gigabyte model download as a side effect.
"""

from __future__ import annotations

from functools import cache
from importlib.util import find_spec


@cache
def is_installed(module: str) -> bool:
    """Return ``True`` when ``module`` can be imported.

    ``find_spec`` raises for a missing *parent* package, which we treat the same
    as "not installed".
    """
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def require(module: str, feature: str) -> None:
    """Raise a helpful error when an optional dependency is missing."""
    if not is_installed(module):
        from ..core.errors import ModelUnavailableError

        raise ModelUnavailableError(
            f"{feature} requires the optional dependency '{module}', "
            "which is not installed in this image.",
            details={"module": module},
        )
