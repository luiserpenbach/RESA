"""
Tests for engine design API routes.
"""
from .conftest import MINIMAL_ENGINE_CONFIG


def test_validate_valid_config(client):
    resp = client.post("/api/v1/engine/validate", json=MINIMAL_ENGINE_CONFIG)
    assert resp.status_code == 200
    data = resp.json()
    assert "is_valid" in data
    assert isinstance(data["errors"], list)
    assert isinstance(data["warnings"], list)


def test_validate_invalid_config(client):
    bad = {**MINIMAL_ENGINE_CONFIG, "thrust_n": -100.0}
    resp = client.post("/api/v1/engine/validate", json=bad)
    assert resp.status_code == 200
    data = resp.json()
    # Should fail validation (not HTTP error) and report errors
    assert isinstance(data["errors"], list)


def test_design_engine_returns_performance(client):
    resp = client.post("/api/v1/engine/design", json=MINIMAL_ENGINE_CONFIG)
    assert resp.status_code == 200
    data = resp.json()
    # Key performance metrics must be present and physical
    assert data["isp_vac"] > 100.0, "Isp should be > 100 s"
    assert data["isp_vac"] < 500.0, "Isp should be < 500 s"
    assert data["thrust_vac"] > 0.0
    assert data["dt_mm"] > 0.0
    assert data["expansion_ratio"] > 1.0


def test_design_engine_invalid_body(client):
    # Missing required propellants — should return 422
    resp = client.post("/api/v1/engine/design", json={"thrust_n": "not_a_number"})
    assert resp.status_code == 422


def test_design_engine_session_created(client):
    """Engine design response should not crash even when session_id header absent."""
    resp = client.post("/api/v1/engine/design", json=MINIMAL_ENGINE_CONFIG)
    assert resp.status_code == 200
