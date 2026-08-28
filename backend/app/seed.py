from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from .database import SessionLocal
from .models import Report

SEED_REPORTS = [
    {
        "patient_name": "Aditya Sharma",
        "age": 22,
        "gender": "Male",
        "phone": "+91 98765 10001",
        "city": "Mumbai",
        "lab_branch": "BKC Flagship",
        "test_type": "Blood Test",
        "doctor_name": "Dr. Naina Rao",
        "status": "ready",
        "priority": "routine",
        "notes": "Annual wellness screening.",
    },
    {
        "patient_name": "Rahul Verma",
        "age": 25,
        "gender": "Male",
        "phone": "+91 98765 10002",
        "city": "Delhi",
        "lab_branch": "Aerocity Express",
        "test_type": "Sugar Test",
        "doctor_name": "Dr. Arvind Menon",
        "status": "processing",
        "priority": "urgent",
        "notes": "Fasting sample received.",
    },
    {
        "patient_name": "Sneha Patil",
        "age": 28,
        "gender": "Female",
        "phone": "+91 98765 10003",
        "city": "Pune",
        "lab_branch": "Koregaon Park",
        "test_type": "Cholesterol",
        "doctor_name": "Dr. Meera Iyer",
        "status": "review",
        "priority": "routine",
        "notes": "Doctor review requested before release.",
    },
    {
        "patient_name": "Priya Mehta",
        "age": 31,
        "gender": "Female",
        "phone": "+91 98765 10004",
        "city": "Ahmedabad",
        "lab_branch": "SG Highway",
        "test_type": "Thyroid",
        "doctor_name": "Dr. Kavya Shah",
        "status": "collected",
        "priority": "routine",
        "notes": "Courier pickup scheduled for evening batch.",
    },
    {
        "patient_name": "Arjun Das",
        "age": 19,
        "gender": "Male",
        "phone": "+91 98765 10005",
        "city": "Kolkata",
        "lab_branch": "Salt Lake",
        "test_type": "CBC Panel",
        "doctor_name": "Dr. Rohan Sen",
        "status": "registered",
        "priority": "urgent",
        "notes": "Walk-in patient.",
    },
]


def seed_database(db: Session) -> int:
    if db.query(Report).count():
        return 0

    now = datetime.now(UTC).replace(tzinfo=None)
    for index, payload in enumerate(SEED_REPORTS):
        report = Report(
            **payload,
            sample_collected_at=now - timedelta(hours=index * 5) if index < 4 else None,
            result_due_at=now + timedelta(hours=8 + index * 6),
        )
        db.add(report)
    db.commit()
    return len(SEED_REPORTS)


def main() -> None:
    db = SessionLocal()
    try:
        created = seed_database(db)
        print(f"Seeded {created} report(s).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
