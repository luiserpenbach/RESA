"""
Tests for torch igniter design API routes.
"""


def test_design_igniter_defaults(client):
    """POST with empty body uses defaults and returns a valid result."""
    resp = client.post("/api/v1/igniter/design", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["combustion"]["flame_temperature_k"] > 2000.0
    assert data["combustion"]["c_star_m_s"] > 500.0
    assert data["geometry"]["chamber_diameter_mm"] > 0.0
    assert data["geometry"]["throat_diameter_mm"] > 0.0
    assert data["performance"]["isp_theoretical_s"] > 100.0
    assert data["performance"]["thrust_n"] > 0.0


import pytest


def test_design_igniter_custom_config(client):
    resp = client.post(
        "/api/v1/igniter/design",
        json={
            "chamber_pressure_pa": 15e5,
            "mixture_ratio": 1.8,
            "total_mass_flow_kg_s": 0.030,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mass_flows"]["total_kg_s"] == pytest.approx(0.030, rel=0.05)


def test_design_igniter_no_body(client):
    """POST with no body (None) should use defaults."""
    resp = client.post("/api/v1/igniter/design")
    assert resp.status_code == 200
