"""
Tests for Monte Carlo uncertainty analysis API routes.
"""
import pytest

from .conftest import MINIMAL_ENGINE_CONFIG


@pytest.fixture()
def session_id(client):
    """Create a session with engine design completed."""
    resp = client.post("/api/v1/session/create", json=MINIMAL_ENGINE_CONFIG)
    assert resp.status_code == 200
    return resp.json()["session_id"]


def test_run_monte_carlo_defaults(client, session_id):
    resp = client.post(
        f"/api/v1/monte-carlo/run?session_id={session_id}",
        json={},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_samples"] > 0
    assert isinstance(data["n_failed"], int)
    assert isinstance(data["statistics"], dict)
    assert isinstance(data["sensitivity"], dict)


def test_run_monte_carlo_custom_params(client, session_id):
    payload = {
        "parameters": [
            {"name": "pc_bar", "nominal": 25.0, "distribution": "normal", "std_dev": 0.75},
            {"name": "mr", "nominal": 4.0, "distribution": "normal", "std_dev": 0.12},
        ],
        "n_samples": 20,
        "output_names": ["combustion.isp_vac"],
    }
    resp = client.post(
        f"/api/v1/monte-carlo/run?session_id={session_id}",
        json=payload,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_samples"] == 20
    assert "combustion.isp_vac" in data["statistics"]


def test_run_monte_carlo_invalid_param(client, session_id):
    payload = {
        "parameters": [
            {"name": "nonexistent_param", "nominal": 1.0, "distribution": "normal", "std_dev": 0.1}
        ],
        "n_samples": 5,
    }
    resp = client.post(
        f"/api/v1/monte-carlo/run?session_id={session_id}",
        json=payload,
    )
    assert resp.status_code == 400
    assert "not supported" in resp.json()["detail"]


def test_run_monte_carlo_missing_session(client):
    resp = client.post(
        "/api/v1/monte-carlo/run?session_id=bad-session-id",
        json={},
    )
    assert resp.status_code == 404


def test_run_monte_carlo_no_session_id(client):
    resp = client.post("/api/v1/monte-carlo/run", json={})
    assert resp.status_code == 422  # session_id is required
