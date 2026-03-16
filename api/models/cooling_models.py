"""
Pydantic models for cooling design API.
"""
from __future__ import annotations

from typing import List, Literal, Optional

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

    # Wall geometry overrides (None = use EngineConfig value)
    wall_thickness_mm: Optional[float] = None
    rib_width_throat_mm: Optional[float] = None

    # Axial margins for CAD import
    start_margin_mm: float = 0.0
    end_margin_mm: float = 0.0

    # Surface roughness override (None = use EngineConfig value)
    roughness_microns: Optional[float] = None

    # Helix/spiral angle
    helix_angle_deg: float = 0.0

    # Coolant inlet overrides (None = use EngineConfig value)
    coolant_p_in_bar: Optional[float] = None
    coolant_t_in_k: Optional[float] = None
    coolant_mass_fraction: Optional[float] = None


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
    wall_thickness_mm_val: float  # effective wall thickness used

    # Arrays as lists for JSON
    x_mm: list[float]
    channel_width_mm: list[float]
    channel_height_mm: list[float]
    rib_width_mm: list[float]
    inner_radius_mm: list[float]

    # Pre-rendered figures
    figure_3d: Optional[str] = None


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
    x_mm: List[float]
    t_wall_hot_k: List[float]
    t_wall_cold_k: List[float]
    t_coolant_k: List[float]
    q_flux_mw_m2: List[float]
    coolant_velocity_m_s: List[float]
    coolant_pressure_bar: List[float]

    warnings: List[str] = []

    # --- N2O-specific extended fields (None when coolant is not N2O) ---
    is_n2o_analysis: bool = False
    min_chf_margin: Optional[float] = None
    max_quality: Optional[float] = None
    # Per-station arrays
    chf_margin: Optional[List[float]] = None
    vapor_quality: Optional[List[float]] = None
    flow_regime: Optional[List[str]] = None
    h_conv_kw_m2k: Optional[List[float]] = None
    density_kg_m3: Optional[List[float]] = None
    # Phase diagram figures
    figure_t_rho: Optional[str] = None
    figure_p_t: Optional[str] = None
