from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.database import Base, get_db
from app.main import app


engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_create_list_update_dashboard_and_delete_report():
    payload = {
        "patient_name": "Asha Nair",
        "age": 34,
        "test_type": "CBC Panel",
        "city": "Mumbai",
        "lab_branch": "BKC Flagship",
        "status": "registered",
        "priority": "urgent",
    }

    created = client.post("/api/reports", json=payload)
    assert created.status_code == 201
    report = created.json()
    assert report["id"]
    assert report["patient_name"] == "Asha Nair"

    listed = client.get("/api/reports", params={"search": "Asha"})
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    updated = client.put(
        f"/api/reports/{report['id']}",
        json={"status": "ready", "priority": "routine"},
    )
    assert updated.status_code == 200
    assert updated.json()["status"] == "ready"

    dashboard = client.get("/api/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["total_reports"] >= 1

    deleted = client.delete(f"/api/reports/{report['id']}")
    assert deleted.status_code == 204
