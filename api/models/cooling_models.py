"""
Pydantic models for cooling design API.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class CoolingChannelConfigRequest(BaseModel):
    """Request model for cooling channel design."""

    channel_type: Literal["rectangular", "trapezoidal"] = "rectangular"
    num_channels_override: Optional[int] = None

    height_profile: Literal["constant", "tapered", "custom"] = "constant"
    height_throat_m: float = 0.75e-3
    height_chamber_m: float = 1.5e-3
    height_exit_m: float = 1.0e-3

    taper_angle_deg: float = 10.0

    bifurcation_enabled: bool = False
    bifurcation_station_x: float = 0.0

    aspect_ratio_limit: float = 10.0
    optimization_target: Literal[
        "none", "min_dp", "min_wall_temp", "max_margin"
    ] = "none"


class CoolingChannelResponse(BaseModel):
    """Response with cooling channel geometry details."""

    num_channels: int
    channel_type: str
    height_profile: str

    # Summary metrics
    min_channel_width_mm: float
    max_channel_width_mm: float
    min_aspect_ratio: float
    max_aspect_ratio: float

    # Arrays as lists for JSON
    x_mm: list[float]
    channel_width_mm: list[float]
    channel_height_mm: list[float]


class CoolingAnalysisResponse(BaseModel):
    """Response from cooling thermal analysis."""

    # Summary metrics
    max_wall_temp_k: float
    max_heat_flux_mw_m2: float
    pressure_drop_bar: float
    outlet_temp_k: float

    # Plotly figure JSON
    figure_thermal: Optional[str] = None

    # Profile arrays
    x_mm: list[float]
    t_wall_hot_k: list[float]
    t_wall_cold_k: list[float]
    t_coolant_k: list[float]
    q_flux_mw_m2: list[float]
    coolant_velocity_m_s: list[float]
    coolant_pressure_bar: list[float]

    warnings: list[str] = []
