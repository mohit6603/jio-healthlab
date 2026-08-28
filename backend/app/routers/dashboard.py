"""Operations dashboard endpoint."""

from fastapi import APIRouter

from ..dependencies import DbSession, Requires
from ..schemas.dashboard import DashboardSummary
from ..security import Permission
from ..services import dashboard_service

router = APIRouter(prefix="/api", tags=["Dashboard"])


@router.get(
    "/dashboard",
    response_model=DashboardSummary,
    dependencies=[Requires(Permission.DASHBOARD_READ)],
    summary="Operational snapshot",
    description="Totals, status/test/city breakdowns, due-soon queue and the "
    "most recent reports.",
)
def dashboard(db: DbSession) -> DashboardSummary:
    return dashboard_service.build_summary(db)
