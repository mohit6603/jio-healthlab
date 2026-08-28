"""System endpoint tests."""


def test_health_reports_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "backend"
    assert body["version"]


def test_health_echoes_request_id(client):
    response = client.get("/health", headers={"X-Request-ID": "trace-abc-123"})

    assert response.headers["X-Request-ID"] == "trace-abc-123"


def test_health_generates_request_id_when_absent(client):
    response = client.get("/health")

    assert response.headers.get("X-Request-ID")


def test_root_banner(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "app" in response.json()


def test_test_types_catalogue(client):
    response = client.get("/api/test-types")

    assert response.status_code == 200
    types = response.json()
    assert "CBC Panel" in types
    assert "Thyroid" in types
