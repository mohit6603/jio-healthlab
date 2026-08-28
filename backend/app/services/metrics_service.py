"""Building the Prometheus exposition payload."""

from __future__ import annotations

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from ..models import AIQueryLog, AuditLog, Report
from ..routers.system import APP_VERSION


def _escape(value: str) -> str:
    """Escape a Prometheus label value."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


class MetricWriter:
    """Accumulates lines in Prometheus text format."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def metric(
        self,
        name: str,
        value: float | int,
        *,
        help_text: str | None = None,
        metric_type: str = "gauge",
        labels: dict[str, str] | None = None,
        declare: bool = True,
    ) -> None:
        if declare:
            if help_text:
                self._lines.append(f"# HELP {name} {help_text}")
            self._lines.append(f"# TYPE {name} {metric_type}")

        rendered = ""
        if labels:
            pairs = ",".join(
                f'{key}="{_escape(str(val))}"' for key, val in sorted(labels.items())
            )
            rendered = "{" + pairs + "}"
        self._lines.append(f"{name}{rendered} {value}")

    def render(self) -> str:
        return "\n".join(self._lines) + "\n"


def render_prometheus(db: Session) -> str:
    """Compute every series from the database."""
    writer = MetricWriter()

    writer.metric(
        "healthlab_build_info",
        1,
        help_text="Build metadata; the value is always 1.",
        labels={"version": APP_VERSION},
    )

    # ── reports ───────────────────────────────────────────────────
    total = db.scalar(select(func.count(Report.id))) or 0
    writer.metric(
        "healthlab_reports_total",
        total,
        help_text="Reports currently stored.",
    )

    status_rows = db.execute(
        select(Report.status, func.count(Report.id)).group_by(Report.status)
    ).all()
    for index, (status, count) in enumerate(status_rows):
        writer.metric(
            "healthlab_reports_by_status",
            int(count),
            help_text="Reports grouped by workflow status.",
            labels={"status": str(status or "unknown")},
            declare=index == 0,
        )

    priority_rows = db.execute(
        select(Report.priority, func.count(Report.id)).group_by(Report.priority)
    ).all()
    for index, (priority, count) in enumerate(priority_rows):
        writer.metric(
            "healthlab_reports_by_priority",
            int(count),
            help_text="Reports grouped by priority.",
            labels={"priority": str(priority or "unknown")},
            declare=index == 0,
        )

    # ── AI usage ──────────────────────────────────────────────────
    ai_rows = db.execute(
        select(
            AIQueryLog.query_type,
            func.count(AIQueryLog.id),
            func.sum(case((AIQueryLog.success.is_(False), 1), else_=0)),
            func.avg(AIQueryLog.latency_ms),
            func.sum(case((AIQueryLog.grounded.is_(False), 1), else_=0)),
        ).group_by(AIQueryLog.query_type)
    ).all()

    for index, (query_type, count, failures, latency, ungrounded) in enumerate(ai_rows):
        labels = {"query_type": str(query_type)}
        writer.metric(
            "healthlab_ai_queries_total",
            int(count or 0),
            help_text="AI interactions, by type.",
            metric_type="counter",
            labels=labels,
            declare=index == 0,
        )
        writer.metric(
            "healthlab_ai_query_failures_total",
            int(failures or 0),
            help_text="AI interactions that returned an error.",
            metric_type="counter",
            labels=labels,
            declare=index == 0,
        )
        writer.metric(
            "healthlab_ai_query_latency_ms_avg",
            round(float(latency), 2) if latency else 0.0,
            help_text="Mean end-to-end latency of AI interactions.",
            labels=labels,
            declare=index == 0,
        )
        writer.metric(
            "healthlab_ai_ungrounded_total",
            int(ungrounded or 0),
            help_text=(
                "AI answers not backed by retrieved material -- refusals, "
                "empty context, or a blocked clinical question."
            ),
            metric_type="counter",
            labels=labels,
            declare=index == 0,
        )

    # ── audit ─────────────────────────────────────────────────────
    audit_rows = db.execute(
        select(AuditLog.action, AuditLog.status, func.count(AuditLog.id)).group_by(
            AuditLog.action, AuditLog.status
        )
    ).all()
    for index, (action, status, count) in enumerate(audit_rows):
        writer.metric(
            "healthlab_audit_events_total",
            int(count),
            help_text="Audited actions, by action and outcome.",
            metric_type="counter",
            labels={"action": str(action), "status": str(status)},
            declare=index == 0,
        )

    return writer.render()
