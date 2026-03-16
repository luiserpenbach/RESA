"""
Pydantic models for tank simulation API.
"""
from __future__ import annotations

from pydantic import BaseModel


class TankConfigRequest(BaseModel):
    volume: float = 0.020
    initial_liquid_mass: float = 15.0
    initial_ullage_pressure: float = 60e5
    initial_temperature: float = 293.15
    ambient_temperature: float = 293.15
    heat_transfer_coefficient: float = 5.0


class PressurantConfigRequest(BaseModel):
    fluid_name: str = "Nitrogen"
    supply_pressure: float = 65e5
    supply_temperature: float = 293.15
    regulator_flow_coefficient: float = 1e-6


class PropellantConfigRequest(BaseModel):
    fluid_name: str = "NitrousOxide"
    mass_flow_rate: float = 0.8
    is_self_pressurizing: bool = True


class TankSimConfigRequest(BaseModel):
    tank_type: str = "n2o"
    tank: TankConfigRequest = TankConfigRequest()
    pressurant: PressurantConfigRequest = PressurantConfigRequest()
    propellant: PropellantConfigRequest = PropellantConfigRequest()
    duration_s: float = 30.0


class TankSimResponse(BaseModel):
    time_s: list[float]
    pressure_bar: list[float]
    liquid_mass_kg: list[float]
    liquid_temperature_k: list[float]
    ullage_temperature_k: list[float]
    burn_duration_s: float
    final_liquid_mass_kg: float
    final_pressure_bar: float
