"""Operational metrics.

Exposed in Prometheus text format, which is the lingua franca: a Prometheus
scrape, an OpenTelemetry collector and a CloudWatch agent can all read it, so
this works for the EC2 deployment and for a future ECS one without change.

Counters are derived from the audit tables rather than kept in process memory.
That keeps them correct across restarts and across replicas, at the cost of a
few aggregate queries per scrape -- the right trade for this volume.
"""

from __future__ import annotations

from fastapi import APIRouter, Response

from ..dependencies import DbSession, Requires
from ..security import Permission
from ..services import metrics_service

router = APIRouter(tags=["System"])

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@router.get(
    "/metrics",
    dependencies=[Requires(Permission.ADMIN_USERS)],
    summary="Prometheus metrics",
    description=(
        "Operational counters in Prometheus text format.\n\n"
        "Administrators only: the series expose usage patterns and error "
        "rates. In a deployment where the scraper cannot authenticate, put "
        "this behind network policy instead and remove the permission "
        "requirement deliberately, not by accident.\n\n"
        "Values are computed from the audit tables, so they survive restarts "
        "and are consistent across replicas."
    ),
    response_class=Response,
    responses={200: {"content": {"text/plain": {}}}},
)
def metrics(db: DbSession) -> Response:
    return Response(
        content=metrics_service.render_prometheus(db), media_type=CONTENT_TYPE
    )
