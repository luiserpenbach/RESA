"""
Pydantic models for engine design API.
Mirrors resa.core.config.EngineConfig and resa.core.results.EngineDesignResult.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict


class EngineConfigRequest(BaseModel):
    """Request model mirroring EngineConfig dataclass field-for-field."""

    model_config = ConfigDict(populate_by_name=True)

    # === IDENTIFICATION ===
    engine_name: str = "Unnamed Engine"
    version: str = "1.0"
    designer: str = ""
    description: str = ""

    # === PROPELLANTS ===
    fuel: str = "Ethanol90"
    oxidizer: str = "N2O"
    fuel_injection_temp_k: float = 298.0
    oxidizer_injection_temp_k: float = 278.0

    # === PERFORMANCE TARGETS ===
    thrust_n: float = 1000.0
    pc_bar: float = 20.0
    mr: float = 4.0
    eff_combustion: float = 0.95

    # === EFFICIENCIES ===
    eff_nozzle_divergence: float = 0.983
    freeze_at_throat: bool = False

    # === NOZZLE DESIGN ===
    nozzle_type: Literal["bell", "conical", "ideal"] = "bell"
    throat_diameter: float = 0.0
    expansion_ratio: float = 0.0
    p_exit_bar: float = 1.013
    L_star: float = 1100.0
    contraction_ratio: float = 10.0
    theta_convergent: float = 30.0
    theta_exit: float = 15.0
    bell_fraction: float = 0.8

    # === COOLING SYSTEM ===
    coolant_name: str = "REFPROP::NitrousOxide"
    cooling_mode: Literal["counter-flow", "co-flow"] = "counter-flow"
    coolant_mass_fraction: float = 1.0
    coolant_p_in_bar: float = 50.0
    coolant_t_in_k: float = 290.0
    channel_width_throat: float = 1.0e-3
    channel_height: float = 0.75e-3
    rib_width_throat: float = 0.6e-3
    wall_thickness: float = 0.5e-3
    wall_roughness: float = 20e-6
    wall_conductivity: float = 15.0
    wall_material: str = "inconel718"

    # === INJECTOR ===
    injector_dp_bar: float = 0.0


class ValidationResponse(BaseModel):
    is_valid: bool
    errors: list[str]
    warnings: list[str]


class CombustionResultResponse(BaseModel):
    pc_bar: float
    mr: float
    cstar: float
    isp_vac: float
    isp_opt: float
    T_combustion: float
    gamma: float
    mw: float
    mach_exit: float


class EngineDesignResponse(BaseModel):
    """Response model for engine design. Plotly figures are serialized as JSON strings."""

    timestamp: str
    run_type: str

    # Operating point
    pc_bar: float
    mr: float

    # Performance
    isp_vac: float
    isp_sea: float
    thrust_vac: float
    thrust_sea: float
    massflow_total: float
    massflow_ox: float
    massflow_fuel: float

    # Key dimensions [mm]
    dt_mm: float
    de_mm: float
    length_mm: float
    expansion_ratio: float

    # Combustion sub-result
    combustion: Optional[CombustionResultResponse] = None

    # Plotly figures as fig.to_json() strings
    figure_dashboard: Optional[str] = None
    figure_contour: Optional[str] = None
    figure_gas_dynamics: Optional[str] = None
    figure_3d: Optional[str] = None

    # Cooling summary (populated when with_cooling=True)
    max_wall_temp: Optional[float] = None
    max_heat_flux: Optional[float] = None
    pressure_drop_bar: Optional[float] = None
    outlet_temp_k: Optional[float] = None

    # Nozzle contour arrays for CSV export
    contour_x_mm: Optional[list[float]] = None
    contour_y_mm: Optional[list[float]] = None

    warnings: list[str] = []
