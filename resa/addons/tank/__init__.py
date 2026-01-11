"""
Tank pressure simulation module for bi-propellant rocket engines.

This module provides simulation capabilities for tank depletion with
correct physics for two-phase self-pressurizing systems (N2O) and
single-phase pressurized tanks (Ethanol).

Example usage:
    from resa.addons.tank import (
        TankConfig, PressurantConfig, PropellantConfig,
        TwoPhaseNitrousTank, EthanolTank
    )

    # Configure N2O tank
    tank_config = TankConfig(
        volume=0.05,  # 50 L
        initial_liquid_mass=36.0,  # kg
        initial_ullage_pressure=100e5,  # 100 bar
        initial_temperature=293.15,  # 20C
        wall_material_properties={},
        ambient_temperature=293.15,
        heat_transfer_coefficient=10.0
    )

    pressurant_config = PressurantConfig(
        fluid_name='Nitrogen',
        supply_pressure=100e5,
        supply_temperature=293.15,
        regulator_flow_coefficient=0.0001
    )

    propellant_config = PropellantConfig(
        fluid_name='NitrousOxide',
        mass_flow_rate=0.6,
        is_self_pressurizing=True
    )

    # Create and run simulation
    tank = TwoPhaseNitrousTank(tank_config, pressurant_config, propellant_config)
    solution = tank.simulate((0, 100))
"""

from .config import TankConfig, PressurantConfig, PropellantConfig
from .simulator import TwoPhaseNitrousTank, EthanolTank
from . import thermodynamics

__all__ = [
    # Configuration dataclasses
    'TankConfig',
    'PressurantConfig',
    'PropellantConfig',
    # Simulator classes
    'TwoPhaseNitrousTank',
    'EthanolTank',
    # Submodules
    'thermodynamics',
]
