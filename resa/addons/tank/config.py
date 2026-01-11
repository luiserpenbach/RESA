"""
Tank simulation configuration dataclasses.

Defines configuration objects for tank, pressurant, and propellant systems.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TankConfig:
    """Configuration for a single pressurized tank.

    Attributes:
        volume: Tank volume in m^3
        initial_liquid_mass: Initial liquid propellant mass in kg
        initial_ullage_pressure: Initial ullage pressure in Pa
        initial_temperature: Initial temperature in K
        wall_material_properties: Dictionary of thermal properties
        ambient_temperature: Ambient temperature in K
        heat_transfer_coefficient: Heat transfer coefficient in W/m^2/K
    """
    volume: float
    initial_liquid_mass: float
    initial_ullage_pressure: float
    initial_temperature: float
    wall_material_properties: Dict[str, Any]
    ambient_temperature: float
    heat_transfer_coefficient: float


@dataclass
class PressurantConfig:
    """Configuration for pressurant gas system.

    Attributes:
        fluid_name: CoolProp fluid name (e.g., 'Nitrogen')
        supply_pressure: Regulated supply pressure in Pa
        supply_temperature: Supply temperature in K
        regulator_flow_coefficient: Flow coefficient in kg/s/Pa for pressure control
    """
    fluid_name: str
    supply_pressure: float
    supply_temperature: float
    regulator_flow_coefficient: float


@dataclass
class PropellantConfig:
    """Configuration for propellant properties.

    Attributes:
        fluid_name: CoolProp fluid name
        mass_flow_rate: Mass flow rate to engine in kg/s
        is_self_pressurizing: True for self-pressurizing propellants (e.g., N2O)
    """
    fluid_name: str
    mass_flow_rate: float
    is_self_pressurizing: bool
