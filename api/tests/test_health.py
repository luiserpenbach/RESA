"""
Tests for health check and basic API availability.
"""


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "RESA" in data["service"]
