"""
Torch Igniter Sizing Tool - RESA Add-on Module

A preliminary design tool for bipropellant torch igniters using
Ethanol and Nitrous Oxide propellants.

Key features:
- NASA CEA integration for combustion calculations
- CoolProp for fluid properties
- L* method chamber sizing
- HEM (Homogeneous Equilibrium Model) injector orifice sizing for N2O
- Performance analysis and operating envelopes

Example:
    >>> from resa.addons.igniter import IgniterDesigner, IgniterConfig
    >>>
    >>> config = IgniterConfig(
    ...     chamber_pressure=20e5,
    ...     mixture_ratio=2.0,
    ...     total_mass_flow=0.050,
    ...     ethanol_feed_pressure=25e5,
    ...     n2o_feed_pressure=30e5,
    ...     ethanol_feed_temperature=298.15,
    ...     n2o_feed_temperature=298.15
    ... )
    >>>
    >>> designer = IgniterDesigner()
    >>> results = designer.design(config)
    >>> print(results.summary())
"""

__version__ = "0.2.0"

# Main configuration and results classes
from .config import IgniterConfig, IgniterResults

# Main designer class
from .designer import IgniterDesigner

# Component modules
from .cea_interface import CEACalculator, estimate_heat_power
from .fluids import FluidProperties, get_ethanol_density, get_n2o_density
from .chamber import ChamberDesigner, size_chamber_from_mass_flow, validate_chamber_design
from .nozzle import NozzleDesigner, size_nozzle_from_mass_flow
from .performance import (
    PerformanceCalculator,
    calculate_all_performance_metrics,
    estimate_ignition_energy,
    calculate_propellant_mass
)
from .injector import InjectorDesigner

# Physical constants (re-exported for convenience)
from .cea_interface import G0, ETHANOL_LHV

__all__ = [
    # Version
    '__version__',

    # Main classes
    'IgniterConfig',
    'IgniterResults',
    'IgniterDesigner',

    # Component classes
    'CEACalculator',
    'FluidProperties',
    'ChamberDesigner',
    'NozzleDesigner',
    'PerformanceCalculator',
    'InjectorDesigner',

    # Convenience functions
    'estimate_heat_power',
    'get_ethanol_density',
    'get_n2o_density',
    'size_chamber_from_mass_flow',
    'validate_chamber_design',
    'size_nozzle_from_mass_flow',
    'calculate_all_performance_metrics',
    'estimate_ignition_energy',
    'calculate_propellant_mass',

    # Constants
    'G0',
    'ETHANOL_LHV',
]
