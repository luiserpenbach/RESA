"""
Tests for design optimization API routes.
"""
import pytest

from .conftest import MINIMAL_ENGINE_CONFIG


@pytest.fixture()
def session_id(client):
    """Create a session with engine design completed."""
    resp = client.post("/api/v1/session/create", json=MINIMAL_ENGINE_CONFIG)
    assert resp.status_code == 200
    return resp.json()["session_id"]


def test_run_optimization_defaults(client, session_id):
    resp = client.post(
        f"/api/v1/optimization/run?session_id={session_id}",
        json={},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["optimal_variables"], dict)
    assert isinstance(data["optimal_outputs"], dict)
    assert "objective_value" in data
    assert isinstance(data["n_evaluations"], int)
    assert data["n_evaluations"] > 0
    assert "converged" in data


def test_run_optimization_single_variable(client, session_id):
    payload = {
        "variables": [
            {"name": "mr", "min_val": 2.0, "max_val": 6.0, "initial": 4.0}
        ],
        "objective": "isp_vac",
        "minimize": False,
        "max_iterations": 20,
        "algorithm": "Nelder-Mead",
    }
    resp = client.post(
        f"/api/v1/optimization/run?session_id={session_id}",
        json=payload,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "mr" in data["optimal_variables"]
    assert data["optimal_variables"]["mr"] >= 2.0
    assert data["optimal_variables"]["mr"] <= 6.0


def test_run_optimization_invalid_variable(client, session_id):
    payload = {
        "variables": [
            {"name": "nonexistent_var", "min_val": 0.0, "max_val": 100.0, "initial": 50.0}
        ],
    }
    resp = client.post(
        f"/api/v1/optimization/run?session_id={session_id}",
        json=payload,
    )
    assert resp.status_code == 400
    assert "not supported" in resp.json()["detail"]


def test_run_optimization_missing_session(client):
    resp = client.post(
        "/api/v1/optimization/run?session_id=bad-session-id",
        json={},
    )
    assert resp.status_code == 404


def test_run_optimization_no_session_id(client):
    resp = client.post("/api/v1/optimization/run", json={})
    assert resp.status_code == 422
