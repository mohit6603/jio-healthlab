"""Pydantic schemas for the operations dashboard."""

from pydantic import BaseModel

from .report import ReportRead


class DashboardSummary(BaseModel):
    """Aggregated operational snapshot for the landing dashboard."""

    total_reports: int
    ready_reports: int
    urgent_reports: int
    avg_age: float | None
    unique_tests: int
    latest_report_id: int | None
    due_soon: list[ReportRead]
    recent_reports: list[ReportRead]
    by_status: dict[str, int]
    by_test_type: dict[str, int]
    by_city: dict[str, int]
