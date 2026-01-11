"""
Swirl Injector Dimensioning Module

A comprehensive toolkit for designing and analyzing swirl coaxial injectors
for liquid rocket engines.

Supports:
- Liquid-Centered Swirl Coaxial (LCSC) injectors
- Gas-Centered Swirl Coaxial (GCSC) injectors
- Cold flow test equivalent calculations

References:
    - Nardi, Rene et al. (2014): "Dimensioning a Simplex Swirl Injector"
    - Anand et al. correlations for GCSC configurations

Example:
    >>> from resa.addons.injector import InjectorConfig, LCSCCalculator
    >>>
    >>> config = InjectorConfig()
    >>> calc = LCSCCalculator(config)
    >>> results = calc.calculate()
    >>> print(results)
"""

from .config import (
    InjectorConfig,
    PropellantConfig,
    OperatingConditions,
    GeometryConfig,
    ColdFlowConfig,
)
from .results import (
    InjectorResults,
    ColdFlowResults,
    InjectorGeometry,
    PerformanceMetrics,
    MassFlowResults,
    FluidProperties,
    PropellantProperties,
)
from .lcsc import LCSCCalculator
from .gcsc import GCSCCalculator
from .cold_flow import ColdFlowCalculator
from .thermodynamics import (
    ThermodynamicCalculator,
    DischargeCoefficients,
    SprayAngleCorrelations,
    FilmThicknessCorrelations,
    calculate_swirl_number,
    calculate_open_area_ratio,
    calculate_choked_mass_flow,
)

__version__ = "2.0.0"
__author__ = "RESA Team"

__all__ = [
    # Configuration
    "InjectorConfig",
    "PropellantConfig",
    "OperatingConditions",
    "GeometryConfig",
    "ColdFlowConfig",
    # Results
    "InjectorResults",
    "ColdFlowResults",
    "InjectorGeometry",
    "PerformanceMetrics",
    "MassFlowResults",
    "FluidProperties",
    "PropellantProperties",
    # Calculators
    "LCSCCalculator",
    "GCSCCalculator",
    "ColdFlowCalculator",
    # Thermodynamics
    "ThermodynamicCalculator",
    "DischargeCoefficients",
    "SprayAngleCorrelations",
    "FilmThicknessCorrelations",
    # Utility functions
    "calculate_swirl_number",
    "calculate_open_area_ratio",
    "calculate_choked_mass_flow",
]


def quick_design(
    mass_flow_fuel: float = 0.20,
    mass_flow_oxidizer: float = 0.80,
    pressure_drop: float = 20e5,
    injector_type: str = "LCSC",
) -> InjectorResults:
    """
    Quick injector design with minimal input.

    Args:
        mass_flow_fuel: Fuel mass flow in kg/s
        mass_flow_oxidizer: Oxidizer mass flow in kg/s
        pressure_drop: Pressure drop across injector in Pa
        injector_type: "LCSC" or "GCSC"

    Returns:
        InjectorResults with complete design

    Example:
        >>> results = quick_design(0.2, 0.8, 20e5)
        >>> print(results.geometry.orifice_diameter)
    """
    config = InjectorConfig(
        operating=OperatingConditions(
            mass_flow_fuel=mass_flow_fuel,
            mass_flow_oxidizer=mass_flow_oxidizer,
            pressure_drop=pressure_drop,
        )
    )

    if injector_type.upper() == "LCSC":
        calc = LCSCCalculator(config)
    elif injector_type.upper() == "GCSC":
        calc = GCSCCalculator(config)
    else:
        raise ValueError(f"Unknown injector type: {injector_type}")

    return calc.calculate()
