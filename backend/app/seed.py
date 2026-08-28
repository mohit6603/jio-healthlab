from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from .config import get_settings
from .database import SessionLocal
from .models import Report
from .security import Role
from .services import auth_service

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


#: Demo accounts, one per role, so RBAC can be exercised immediately.
#: Passwords are development-only and are refused in production by
#: Settings.assert_production_ready().
SEED_USERS = [
    ("admin@jiohealthlab.example.com", "Priya Admin", Role.ADMIN),
    ("tech@jiohealthlab.example.com", "Ravi Technician", Role.LAB_TECH),
    ("doctor@jiohealthlab.example.com", "Dr Meera Iyer", Role.DOCTOR),
    ("viewer@jiohealthlab.example.com", "Sam Viewer", Role.VIEWER),
]


def seed_users(db: Session) -> int:
    """Create the demo accounts if they do not exist. Idempotent."""
    settings = get_settings()
    created = 0

    for email, full_name, role in SEED_USERS:
        if auth_service.get_user_by_email(db, email) is not None:
            continue
        # One shared development password across the demo accounts; the
        # production guard refuses to start if it is still the default.
        auth_service.create_user(
            db,
            email=email,
            full_name=full_name,
            password=settings.seed_admin_password,
            role=role,
        )
        created += 1

    return created


def main() -> None:
    db = SessionLocal()
    try:
        created = seed_database(db)
        print(f"Seeded {created} report(s).")
        users = seed_users(db)
        print(f"Seeded {users} user(s).")
        if users:
            print(
                "  Demo accounts (development only): "
                + ", ".join(email for email, _, _ in SEED_USERS)
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
