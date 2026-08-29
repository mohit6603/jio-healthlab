"""Roles and permissions.

Authorisation is decided in one place. Routes declare the *permission* they
need, never the role that happens to hold it, so changing what a role can do
means editing this table and nothing else -- no string comparisons scattered
through handlers.
"""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """Who someone is."""

    ADMIN = "ADMIN"
    LAB_TECH = "LAB_TECH"
    DOCTOR = "DOCTOR"
    VIEWER = "VIEWER"


class Permission(StrEnum):
    """What someone may do."""

    # Reports
    REPORTS_READ = "reports:read"
    REPORTS_CREATE = "reports:create"
    REPORTS_UPDATE = "reports:update"
    REPORTS_DELETE = "reports:delete"

    # Dashboard
    DASHBOARD_READ = "dashboard:read"

    # AI -- informational, safe for anyone who can see the product
    AI_CHAT = "ai:chat"
    AI_SEARCH = "ai:search"

    # AI -- about a specific patient's request
    AI_EXPLAIN_REPORT = "ai:explain_report"

    # AI -- operational
    AI_RISK_ANALYTICS = "ai:risk_analytics"
    AI_REPORT_SEARCH = "ai:report_search"

    # Administration
    ADMIN_DOCUMENTS = "admin:documents"
    ADMIN_INDEX = "admin:index"
    ADMIN_USERS = "admin:users"
    ADMIN_MODELS = "admin:models"


#: Read-only access plus general informational AI.
_VIEWER: frozenset[Permission] = frozenset(
    {
        Permission.REPORTS_READ,
        Permission.DASHBOARD_READ,
        Permission.AI_CHAT,
        Permission.AI_SEARCH,
    }
)

#: Lab technicians move work through the pipeline and watch the queue.
_LAB_TECH: frozenset[Permission] = _VIEWER | {
    Permission.REPORTS_CREATE,
    Permission.REPORTS_UPDATE,
    Permission.AI_RISK_ANALYTICS,
    Permission.AI_REPORT_SEARCH,
}

#: Doctors read reports and may ask for a test to be explained. They do not
#: move work through the lab pipeline.
_DOCTOR: frozenset[Permission] = _VIEWER | {
    Permission.AI_EXPLAIN_REPORT,
    Permission.AI_REPORT_SEARCH,
}

_ADMIN: frozenset[Permission] = frozenset(Permission)

ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.VIEWER: _VIEWER,
    Role.LAB_TECH: _LAB_TECH,
    Role.DOCTOR: _DOCTOR,
    Role.ADMIN: _ADMIN,
}


def permissions_for(role: Role | str) -> frozenset[Permission]:
    """Permissions granted to ``role``. An unknown role grants nothing."""
    try:
        resolved = Role(role)
    except ValueError:
        return frozenset()
    return ROLE_PERMISSIONS.get(resolved, frozenset())


def has_permission(role: Role | str, permission: Permission) -> bool:
    return permission in permissions_for(role)
