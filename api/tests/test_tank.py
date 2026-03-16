"""
Tests for tank simulation API routes.
"""


def test_simulate_n2o_defaults(client):
    resp = client.post("/api/v1/tank/simulate", json={"tank_type": "n2o"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["burn_duration_s"] > 0.0
    assert len(data["time_s"]) > 0
    assert len(data["pressure_bar"]) == len(data["time_s"])
    assert len(data["liquid_mass_kg"]) == len(data["time_s"])
    assert data["final_liquid_mass_kg"] >= 0.0


def test_simulate_ethanol_defaults(client):
    resp = client.post("/api/v1/tank/simulate", json={"tank_type": "ethanol"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["burn_duration_s"] > 0.0
    assert len(data["time_s"]) > 0


def test_simulate_tank_invalid_type(client):
    resp = client.post("/api/v1/tank/simulate", json={"tank_type": "helium"})
    assert resp.status_code == 400
    assert "Unknown tank_type" in resp.json()["detail"]


def test_simulate_tank_output_downsampled(client):
    """Output should be at most 200 points."""
    resp = client.post(
        "/api/v1/tank/simulate",
        json={"tank_type": "n2o", "duration_s": 60.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["time_s"]) <= 200


def test_simulate_tank_no_body(client):
    resp = client.post("/api/v1/tank/simulate")
    assert resp.status_code == 200
