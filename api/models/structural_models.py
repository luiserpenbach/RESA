"""
Pydantic models for structural / wall thickness API.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class MaterialPropertiesResponse(BaseModel):
    """Material properties for display."""

    name: str
    density_kg_m3: float
    yield_strength_mpa: float
    thermal_conductivity_w_mk: float
    max_service_temp_k: float


class WallThicknessConfigRequest(BaseModel):
    """Request model for wall thickness analysis."""

    material_name: str = "inconel718"
    safety_factor_pressure: float = 2.0
    safety_factor_thermal: float = 1.5
    design_life_cycles: int = 100
    include_fatigue: bool = False


class WallThicknessResponse(BaseModel):
    """Response from wall thickness analysis."""

    # Summary
    min_safety_factor: float
    critical_station_x_mm: float
    max_hoop_stress_mpa: float
    max_thermal_stress_mpa: float
    max_von_mises_mpa: float

    # Material used
    material: MaterialPropertiesResponse

    # Plotly figure JSON
    figure_stress: Optional[str] = None
    figure_safety_factor: Optional[str] = None

    # Profile arrays
    x_mm: list[float]
    min_thickness_pressure_mm: list[float]
    min_thickness_thermal_mm: list[float]
    actual_thickness_mm: list[float]
    safety_factor: list[float]
    hoop_stress_mpa: list[float]
    thermal_stress_mpa: list[float]
    von_mises_mpa: list[float]

    warnings: list[str] = []


class MaterialListResponse(BaseModel):
    """List of available materials."""

    materials: dict[str, str]  # id -> display name
