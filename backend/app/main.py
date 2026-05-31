from datetime import UTC, datetime, timedelta

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from .config import get_settings
from .database import get_db
from .models import Report
from .schemas import DashboardSummary, HealthResponse, ReportCreate, ReportRead, ReportUpdate

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="3.0.0",
    description="Diagnostics report operations API for JIO Healthlab.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEST_TYPES = [
    "Blood Test",
    "Sugar Test",
    "Cholesterol",
    "Thyroid",
    "CBC Panel",
    "Urine Test",
    "X-Ray",
    "MRI Scan",
    "Liver Function",
    "Kidney Function",
]


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {
        "app": settings.app_name,
        "message": "Open /docs for the API or run the React frontend.",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.environment,
    )


@app.get("/api/test-types", response_model=list[str], tags=["reports"])
def get_test_types() -> list[str]:
    return TEST_TYPES


@app.get("/api/reports", response_model=list[ReportRead], tags=["reports"])
def list_reports(
    search: str | None = Query(default=None, max_length=120),
    status_filter: str | None = Query(default=None, alias="status"),
    priority: str | None = Query(default=None),
    city: str | None = Query(default=None),
    test_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[Report]:
    stmt = select(Report)

    if search:
        like = f"%{search.strip()}%"
        search_terms = [
            Report.patient_name.ilike(like),
            Report.test_type.ilike(like),
            Report.city.ilike(like),
            Report.lab_branch.ilike(like),
            Report.doctor_name.ilike(like),
        ]
        if search.isdigit():
            search_terms.append(Report.id == int(search))
        stmt = stmt.where(or_(*search_terms))

    if status_filter and status_filter != "all":
        stmt = stmt.where(Report.status == status_filter)
    if priority and priority != "all":
        stmt = stmt.where(Report.priority == priority)
    if city and city != "all":
        stmt = stmt.where(Report.city == city)
    if test_type and test_type != "all":
        stmt = stmt.where(Report.test_type == test_type)

    stmt = stmt.order_by(Report.created_at.desc(), Report.id.desc()).limit(limit).offset(offset)
    return list(db.execute(stmt).scalars())


@app.get("/api/reports/{report_id}", response_model=ReportRead, tags=["reports"])
def get_report(report_id: int, db: Session = Depends(get_db)) -> Report:
    report = db.get(Report, report_id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return report


@app.post(
    "/api/reports",
    response_model=ReportRead,
    status_code=status.HTTP_201_CREATED,
    tags=["reports"],
)
def create_report(payload: ReportCreate, db: Session = Depends(get_db)) -> Report:
    report = Report(**payload.model_dump())
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


@app.put("/api/reports/{report_id}", response_model=ReportRead, tags=["reports"])
def update_report(report_id: int, payload: ReportUpdate, db: Session = Depends(get_db)) -> Report:
    report = db.get(Report, report_id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(report, field, value)

    db.commit()
    db.refresh(report)
    return report


@app.delete("/api/reports/{report_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["reports"])
def delete_report(report_id: int, db: Session = Depends(get_db)) -> None:
    report = db.get(Report, report_id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    db.delete(report)
    db.commit()
    return None


@app.get("/api/dashboard", response_model=DashboardSummary, tags=["dashboard"])
def dashboard(db: Session = Depends(get_db)) -> DashboardSummary:
    total_reports = db.scalar(select(func.count(Report.id))) or 0
    ready_reports = db.scalar(select(func.count(Report.id)).where(Report.status == "ready")) or 0
    urgent_reports = db.scalar(select(func.count(Report.id)).where(Report.priority == "urgent")) or 0
    avg_age = db.scalar(select(func.avg(Report.age)))
    unique_tests = db.scalar(select(func.count(func.distinct(Report.test_type)))) or 0
    latest_report_id = db.scalar(select(func.max(Report.id)))

    due_cutoff = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=24)
    due_soon = list(
        db.execute(
            select(Report)
            .where(
                Report.result_due_at.is_not(None),
                Report.result_due_at <= due_cutoff,
                Report.status.notin_(["delivered", "ready"]),
            )
            .order_by(Report.result_due_at.asc())
            .limit(6)
        ).scalars()
    )

    recent_reports = list(
        db.execute(select(Report).order_by(Report.created_at.desc()).limit(6)).scalars()
    )

    return DashboardSummary(
        total_reports=total_reports,
        ready_reports=ready_reports,
        urgent_reports=urgent_reports,
        avg_age=round(float(avg_age), 1) if avg_age is not None else None,
        unique_tests=unique_tests,
        latest_report_id=latest_report_id,
        due_soon=due_soon,
        recent_reports=recent_reports,
        by_status=_count_by(db, Report.status),
        by_test_type=_count_by(db, Report.test_type),
        by_city=_count_by(db, Report.city),
    )


def _count_by(db: Session, column) -> dict[str, int]:
    rows = db.execute(select(column, func.count(Report.id)).group_by(column)).all()
    return {str(label or "Unknown"): int(count) for label, count in rows}
