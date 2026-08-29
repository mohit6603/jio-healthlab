"""End-to-end report lifecycle test (preserved from the original suite)."""


def test_create_list_update_dashboard_and_delete_report(client, report_payload):
    created = client.post("/api/reports", json=report_payload)
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
