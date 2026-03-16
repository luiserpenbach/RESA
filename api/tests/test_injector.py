"""
Tests for swirl injector design API routes.
"""


def test_design_injector_lcsc_defaults(client):
    resp = client.post("/api/v1/injector/design", json={"injector_type": "LCSC"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["injector_type"] == "LCSC"
    assert data["geometry"]["fuel_orifice_radius_mm"] > 0.0
    assert data["geometry"]["swirl_chamber_radius_mm"] > 0.0
    assert data["performance"]["discharge_coefficient"] > 0.0
    assert data["mass_flows"]["mixture_ratio"] > 0.0


def test_design_injector_gcsc(client):
    resp = client.post("/api/v1/injector/design", json={"injector_type": "GCSC"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["injector_type"] == "GCSC"


def test_design_injector_case_insensitive(client):
    resp = client.post("/api/v1/injector/design", json={"injector_type": "lcsc"})
    assert resp.status_code == 200


def test_design_injector_invalid_type(client):
    resp = client.post("/api/v1/injector/design", json={"injector_type": "INVALID"})
    assert resp.status_code == 400
    assert "Unknown injector type" in resp.json()["detail"]


def test_design_injector_no_body(client):
    resp = client.post("/api/v1/injector/design")
    assert resp.status_code == 200
