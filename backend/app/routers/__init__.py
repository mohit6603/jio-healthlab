"""HTTP routers grouped by domain."""

from . import ai, audit, auth, dashboard, metrics, reports, system

__all__ = ["ai", "audit", "auth", "dashboard", "metrics", "reports", "system"]
