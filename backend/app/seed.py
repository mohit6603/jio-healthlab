from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from .config import get_settings
from .database import SessionLocal
from .models import Report
from .security import Role
from .services import auth_service

#: Synthetic demo data. Names, contact details and notes are invented; no
#: real patient information appears anywhere in this project.
#:
#: Deliberately spread across branches, cities, test types, priorities and
#: statuses so the dashboard, the delay-risk analytics and semantic report
#: search all have something meaningful to show on a fresh install.
SEED_REPORTS = [
    # Mumbai -- BKC Flagship (well resourced)
    {"patient_name": "Aditya Sharma", "age": 22, "gender": "Male",
     "phone": "+91 98765 10001", "city": "Mumbai", "lab_branch": "BKC Flagship",
     "test_type": "Blood Test", "doctor_name": "Dr. Naina Rao",
     "status": "ready", "priority": "routine",
     "notes": "Annual wellness screening."},
    {"patient_name": "Kavya Menon", "age": 45, "gender": "Female",
     "phone": "+91 98765 10011", "city": "Mumbai", "lab_branch": "BKC Flagship",
     "test_type": "Lipid Profile", "doctor_name": "Dr. Naina Rao",
     "status": "review", "priority": "routine",
     "notes": "Fasting sample, 11 hours."},
    {"patient_name": "Imran Qureshi", "age": 61, "gender": "Male",
     "phone": "+91 98765 10012", "city": "Mumbai", "lab_branch": "BKC Flagship",
     "test_type": "Kidney Function", "doctor_name": "Dr. Sameer Joshi",
     "status": "processing", "priority": "urgent",
     "notes": "Pre-operative workup."},

    # Mumbai -- Andheri Hub (under-resourced; higher predicted risk)
    {"patient_name": "Sneha Patil", "age": 28, "gender": "Female",
     "phone": "+91 98765 10003", "city": "Mumbai", "lab_branch": "Andheri Hub",
     "test_type": "Cholesterol", "doctor_name": "Dr. Meera Iyer",
     "status": "review", "priority": "routine",
     "notes": "Doctor review requested before release."},
    {"patient_name": "Vikram Nair", "age": 54, "gender": "Male",
     "phone": "+91 98765 10013", "city": "Mumbai", "lab_branch": "Andheri Hub",
     "test_type": "CBC Panel", "doctor_name": "Dr. Meera Iyer",
     "status": "processing", "priority": "urgent",
     "notes": "Repeat after haemolysed first draw."},
    {"patient_name": "Anita Desai", "age": 37, "gender": "Female",
     "phone": "+91 98765 10014", "city": "Mumbai", "lab_branch": "Andheri Hub",
     "test_type": "Thyroid", "doctor_name": "Dr. Sameer Joshi",
     "status": "collected", "priority": "routine",
     "notes": "Morning draw for consistent TSH timing."},

    # Delhi
    {"patient_name": "Rahul Verma", "age": 25, "gender": "Male",
     "phone": "+91 98765 10002", "city": "Delhi",
     "lab_branch": "Aerocity Express", "test_type": "Sugar Test",
     "doctor_name": "Dr. Arvind Menon", "status": "processing",
     "priority": "urgent", "notes": "Fasting sample received."},
    {"patient_name": "Farah Sheikh", "age": 33, "gender": "Female",
     "phone": "+91 98765 10015", "city": "Delhi",
     "lab_branch": "Connaught Place", "test_type": "Liver Function",
     "doctor_name": "Dr. Arvind Menon", "status": "registered",
     "priority": "routine", "notes": "Courier pickup scheduled."},

    # Pune
    {"patient_name": "Rohan Kulkarni", "age": 52, "gender": "Male",
     "phone": "+91 98765 10016", "city": "Pune", "lab_branch": "Koregaon Park",
     "test_type": "Thyroid", "doctor_name": "Dr. Kavya Shah",
     "status": "ready", "priority": "routine", "notes": None},
    {"patient_name": "Meera Joshi", "age": 8, "gender": "Female",
     "phone": "+91 98765 10017", "city": "Pune", "lab_branch": "Koregaon Park",
     "test_type": "CBC Panel", "doctor_name": "Dr. Kavya Shah",
     "status": "collected", "priority": "urgent",
     "notes": "Paediatric draw, small volume tube."},

    # Ahmedabad
    {"patient_name": "Priya Mehta", "age": 31, "gender": "Female",
     "phone": "+91 98765 10004", "city": "Ahmedabad",
     "lab_branch": "SG Highway", "test_type": "Urine Test",
     "doctor_name": "Dr. Kavya Shah", "status": "delivered",
     "priority": "routine", "notes": "Report delivered by email."},

    # Kolkata -- Salt Lake (least resourced)
    {"patient_name": "Arjun Das", "age": 19, "gender": "Male",
     "phone": "+91 98765 10005", "city": "Kolkata", "lab_branch": "Salt Lake",
     "test_type": "CBC Panel", "doctor_name": "Dr. Rohan Sen",
     "status": "registered", "priority": "urgent", "notes": "Walk-in patient."},
    {"patient_name": "Sunita Ghosh", "age": 72, "gender": "Female",
     "phone": "+91 98765 10018", "city": "Kolkata", "lab_branch": "Salt Lake",
     "test_type": "Kidney Function", "doctor_name": "Dr. Rohan Sen",
     "status": "processing", "priority": "routine",
     "notes": "Monitoring, six-monthly."},

    # Bengaluru
    {"patient_name": "Nikhil Rao", "age": 41, "gender": "Male",
     "phone": "+91 98765 10019", "city": "Bengaluru",
     "lab_branch": "Whitefield", "test_type": "MRI Scan",
     "doctor_name": "Dr. Latha Krishnan", "status": "processing",
     "priority": "routine", "notes": "Imaging slot confirmed."},
    {"patient_name": "Divya Pillai", "age": 29, "gender": "Female",
     "phone": "+91 98765 10020", "city": "Bengaluru",
     "lab_branch": "Whitefield", "test_type": "X-Ray",
     "doctor_name": "Dr. Latha Krishnan", "status": "ready",
     "priority": "urgent", "notes": None},
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
