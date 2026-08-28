"""Laboratory report ORM model."""

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class Report(Base):
    """A single diagnostic order tracked through the lab workflow."""

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str | None] = mapped_column(String(32), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    city: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    lab_branch: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    test_type: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    doctor_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="registered", index=True
    )
    priority: Mapped[str] = mapped_column(
        String(24), nullable=False, default="routine", index=True
    )
    sample_collected_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    result_due_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, index=True
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Report id={self.id} test_type={self.test_type!r} status={self.status!r}>"
