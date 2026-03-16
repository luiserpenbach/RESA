"""
Shared fixtures for RESA API tests.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Return a TestClient wrapping the FastAPI app."""
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Minimal valid engine config for use across tests
# ---------------------------------------------------------------------------

MINIMAL_ENGINE_CONFIG = {
    "engine_name": "TestEngine",
    "fuel": "Ethanol90",
    "oxidizer": "N2O",
    "thrust_n": 2000.0,
    "pc_bar": 25.0,
    "mr": 4.0,
    "nozzle_efficiency": 0.98,
    "combustion_efficiency": 0.95,
    "expansion_ratio": 6.0,
}
