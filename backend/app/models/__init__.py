"""SQLAlchemy models.

Importing this package registers every table on ``Base.metadata`` so Alembic
autogeneration and ``create_all`` see the complete schema.
"""

from .report import Report

__all__ = ["Report"]
