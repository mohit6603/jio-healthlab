from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


ReportStatus = Literal[
    "registered",
    "collected",
    "processing",
    "review",
    "ready",
    "delivered",
]
ReportPriority = Literal["routine", "urgent"]


class ReportBase(BaseModel):
    patient_name: str = Field(min_length=2, max_length=120)
    age: int = Field(ge=0, le=120)
    test_type: str = Field(min_length=2, max_length=120)
    gender: str | None = Field(default=None, max_length=32)
    phone: str | None = Field(default=None, max_length=32)
    email: EmailStr | None = None
    city: str | None = Field(default=None, max_length=80)
    lab_branch: str | None = Field(default=None, max_length=120)
    doctor_name: str | None = Field(default=None, max_length=120)
    status: ReportStatus = "registered"
    priority: ReportPriority = "routine"
    sample_collected_at: datetime | None = None
    result_due_at: datetime | None = None
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator(
        "patient_name",
        "test_type",
        "gender",
        "phone",
        "city",
        "lab_branch",
        "doctor_name",
        "notes",
        mode="before",
    )
    @classmethod
    def strip_blank_strings(cls, value):
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value


class ReportCreate(ReportBase):
    pass


class ReportUpdate(BaseModel):
    patient_name: str | None = Field(default=None, min_length=2, max_length=120)
    age: int | None = Field(default=None, ge=0, le=120)
    test_type: str | None = Field(default=None, min_length=2, max_length=120)
    gender: str | None = Field(default=None, max_length=32)
    phone: str | None = Field(default=None, max_length=32)
    email: EmailStr | None = None
    city: str | None = Field(default=None, max_length=80)
    lab_branch: str | None = Field(default=None, max_length=120)
    doctor_name: str | None = Field(default=None, max_length=120)
    status: ReportStatus | None = None
    priority: ReportPriority | None = None
    sample_collected_at: datetime | None = None
    result_due_at: datetime | None = None
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator(
        "patient_name",
        "test_type",
        "gender",
        "phone",
        "city",
        "lab_branch",
        "doctor_name",
        "notes",
        mode="before",
    )
    @classmethod
    def strip_blank_strings(cls, value):
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value


class ReportRead(ReportBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DashboardSummary(BaseModel):
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


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
