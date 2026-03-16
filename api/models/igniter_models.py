"""
Pydantic models for torch igniter design API.
"""
from __future__ import annotations

from pydantic import BaseModel


class IgniterConfigRequest(BaseModel):
    chamber_pressure_pa: float = 20e5
    mixture_ratio: float = 2.0
    total_mass_flow_kg_s: float = 0.050
    ethanol_feed_pressure_pa: float = 25e5
    n2o_feed_pressure_pa: float = 30e5
    ethanol_feed_temperature_k: float = 298.15
    n2o_feed_temperature_k: float = 298.15
    l_star: float = 1.0
    expansion_ratio: float = 3.0
    nozzle_type: str = "conical"
    n2o_orifice_count: int = 4
    ethanol_orifice_count: int = 4
    discharge_coefficient: float = 0.7


class IgniterCombustionResponse(BaseModel):
    flame_temperature_k: float
    c_star_m_s: float
    gamma: float
    molecular_weight: float
    heat_power_kw: float


class IgniterGeometryResponse(BaseModel):
    chamber_diameter_mm: float
    chamber_length_mm: float
    chamber_volume_cm3: float
    throat_diameter_mm: float
    exit_diameter_mm: float
    nozzle_length_mm: float


class IgniterInjectorResponse(BaseModel):
    n2o_orifice_diameter_mm: float
    ethanol_orifice_diameter_mm: float
    n2o_injection_velocity_m_s: float
    ethanol_injection_velocity_m_s: float
    n2o_pressure_drop_bar: float
    ethanol_pressure_drop_bar: float


class IgniterPerformanceResponse(BaseModel):
    isp_theoretical_s: float
    thrust_n: float


class IgniterMassFlowsResponse(BaseModel):
    total_kg_s: float
    oxidizer_kg_s: float
    fuel_kg_s: float
    mixture_ratio: float


class IgniterResponse(BaseModel):
    combustion: IgniterCombustionResponse
    geometry: IgniterGeometryResponse
    injector: IgniterInjectorResponse
    performance: IgniterPerformanceResponse
    mass_flows: IgniterMassFlowsResponse
