"""
Pydantic models for swirl injector design API.
"""
from __future__ import annotations

from pydantic import BaseModel


class PropellantConfigRequest(BaseModel):
    fuel: str = "REFPROP::Ethanol"
    oxidizer: str = "REFPROP::NitrousOxide"
    fuel_temperature: float = 300.0
    oxidizer_temperature: float = 500.0


class OperatingConditionsRequest(BaseModel):
    inlet_pressure: float = 45e5
    pressure_drop: float = 20e5
    mass_flow_fuel: float = 0.20
    mass_flow_oxidizer: float = 0.80
    oxidizer_velocity: float = 100.0


class GeometryConfigRequest(BaseModel):
    num_elements: int = 3
    num_fuel_ports: int = 3
    num_ox_orifices: int = 1
    post_thickness: float = 0.5e-3
    spray_half_angle: float = 60.0
    minimum_clearance: float = 0.5e-3


class InjectorConfigRequest(BaseModel):
    injector_type: str = "LCSC"
    propellants: PropellantConfigRequest = PropellantConfigRequest()
    operating: OperatingConditionsRequest = OperatingConditionsRequest()
    geometry: GeometryConfigRequest = GeometryConfigRequest()


class InjectorGeometryResponse(BaseModel):
    fuel_orifice_radius_mm: float
    fuel_port_radius_mm: float
    swirl_chamber_radius_mm: float
    ox_outlet_radius_mm: float
    ox_inlet_orifice_radius_mm: float
    recess_length_mm: float


class InjectorPerformanceResponse(BaseModel):
    spray_half_angle_deg: float
    swirl_number: float
    momentum_flux_ratio: float
    velocity_ratio: float
    weber_number: float
    discharge_coefficient: float


class InjectorMassFlowResponse(BaseModel):
    fuel_per_element_kg_s: float
    oxidizer_per_element_kg_s: float
    total_fuel_kg_s: float
    total_oxidizer_kg_s: float
    mixture_ratio: float


class InjectorResponse(BaseModel):
    injector_type: str
    geometry: InjectorGeometryResponse
    performance: InjectorPerformanceResponse
    mass_flows: InjectorMassFlowResponse
