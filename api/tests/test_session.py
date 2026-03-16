"""
Tests for session management API routes.
"""
from .conftest import MINIMAL_ENGINE_CONFIG


def test_create_session(client):
    resp = client.post("/api/v1/session/create", json=MINIMAL_ENGINE_CONFIG)
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert len(data["session_id"]) > 0
    assert "module_status" in data


def test_get_session_status(client):
    create_resp = client.post("/api/v1/session/create", json=MINIMAL_ENGINE_CONFIG)
    assert create_resp.status_code == 200
    session_id = create_resp.json()["session_id"]

    status_resp = client.get(f"/api/v1/session/{session_id}/status")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["session_id"] == session_id
    assert "module_status" in data


def test_get_session_not_found(client):
    resp = client.get("/api/v1/session/nonexistent-session-id/status")
    assert resp.status_code == 404


def test_delete_session(client):
    create_resp = client.post("/api/v1/session/create", json=MINIMAL_ENGINE_CONFIG)
    session_id = create_resp.json()["session_id"]

    del_resp = client.delete(f"/api/v1/session/{session_id}")
    assert del_resp.status_code == 200

    # Now it should be gone
    status_resp = client.get(f"/api/v1/session/{session_id}/status")
    assert status_resp.status_code == 404


def test_delete_nonexistent_session(client):
    resp = client.delete("/api/v1/session/does-not-exist")
    assert resp.status_code == 404
