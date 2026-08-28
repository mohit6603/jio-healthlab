"""Pydantic schemas for laboratory reports."""

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

# Fields that are trimmed and coerced to NULL when blank.
_TRIMMED_FIELDS = (
    "patient_name",
    "test_type",
    "gender",
    "phone",
    "city",
    "lab_branch",
    "doctor_name",
    "notes",
)


class _TrimBlankStrings(BaseModel):
    """Mixin: strip surrounding whitespace and treat empty strings as NULL."""

    @field_validator(*_TRIMMED_FIELDS, mode="before", check_fields=False)
    @classmethod
    def strip_blank_strings(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value


class ReportBase(_TrimBlankStrings):
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


class ReportCreate(ReportBase):
    """Payload for ``POST /api/reports``."""


class ReportUpdate(_TrimBlankStrings):
    """Partial update payload for ``PUT /api/reports/{id}``."""

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


class ReportRead(ReportBase):
    """Full report representation returned by the API."""

    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ReportFilters(BaseModel):
    """Normalised query parameters for the report list endpoint."""

    search: str | None = None
    status: str | None = None
    priority: str | None = None
    city: str | None = None
    test_type: str | None = None
    limit: int = 100
    offset: int = 0
