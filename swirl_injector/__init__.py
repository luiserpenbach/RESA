"""
Swirl Injector Dimensioning Tool

A comprehensive toolkit for designing and analyzing swirl coaxial injectors
for liquid rocket engines.

Supports:
- Liquid-Centered Swirl Coaxial (LCSC) injectors
- Gas-Centered Swirl Coaxial (GCSC) injectors
- Cold flow test equivalent calculations
- Sensitivity analysis and visualization

References:
    - Nardi, Rene et al. (2014): "Dimensioning a Simplex Swirl Injector"
    - Anand et al. correlations for GCSC configurations

Example:
    >>> from swirl_injector import InjectorConfig, LCSCCalculator
    >>> 
    >>> config = InjectorConfig()
    >>> calc = LCSCCalculator(config)
    >>> results = calc.calculate()
    >>> print(results)
"""

from config import (
    InjectorConfig,
    PropellantConfig,
    OperatingConditions,
    GeometryConfig,
    ColdFlowConfig
)
from results import (
    InjectorResults,
    ColdFlowResults,
    InjectorGeometry,
    PerformanceMetrics,
    MassFlowResults,
    FluidProperties,
    PropellantProperties
)
from calculators import (
    LCSCCalculator,
    GCSCCalculator,
    ColdFlowCalculator
)
from thermodynamics import (
    ThermodynamicCalculator,
    DischargeCoefficients,
    SprayAngleCorrelations,
    FilmThicknessCorrelations
)
from visualization import (
    plot_injector_cross_section,
    plot_discharge_coefficient_comparison,
    plot_spray_angle_comparison,
    plot_sensitivity_analysis,
    plot_operating_envelope,
    save_figure
)

__version__ = "2.0.0"
__author__ = "Refactored Swirl Injector Team"

__all__ = [
    # Configuration
    'InjectorConfig',
    'PropellantConfig',
    'OperatingConditions',
    'GeometryConfig',
    'ColdFlowConfig',
    
    # Results
    'InjectorResults',
    'ColdFlowResults',
    'InjectorGeometry',
    'PerformanceMetrics',
    'MassFlowResults',
    'FluidProperties',
    'PropellantProperties',
    
    # Calculators
    'LCSCCalculator',
    'GCSCCalculator',
    'ColdFlowCalculator',
    
    # Thermodynamics
    'ThermodynamicCalculator',
    'DischargeCoefficients',
    'SprayAngleCorrelations',
    'FilmThicknessCorrelations',
    
    # Visualization
    'plot_injector_cross_section',
    'plot_discharge_coefficient_comparison',
    'plot_spray_angle_comparison',
    'plot_sensitivity_analysis',
    'plot_operating_envelope',
    'save_figure',
]


def quick_design(
    mass_flow_fuel: float = 0.20,
    mass_flow_oxidizer: float = 0.80,
    pressure_drop: float = 20e5,
    injector_type: str = "LCSC"
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
            pressure_drop=pressure_drop
        )
    )
    
    if injector_type.upper() == "LCSC":
        calc = LCSCCalculator(config)
    elif injector_type.upper() == "GCSC":
        calc = GCSCCalculator(config)
    else:
        raise ValueError(f"Unknown injector type: {injector_type}")
    
    return calc.calculate()


if __name__ == "__main__":
    # Example usage demonstrating the refactored API
    print("=" * 60)
    print("Swirl Injector Dimensioning Tool - Example")
    print("=" * 60)
    
    # Method 1: Using default configuration
    print("\n--- Method 1: Default Configuration ---")
    config = InjectorConfig()
    
    # Method 2: Custom configuration via dataclasses
    print("\n--- Method 2: Custom Configuration ---")
    config = InjectorConfig(
        propellants=PropellantConfig(
            fuel="REFPROP::Ethanol",
            oxidizer="REFPROP::NitrousOxide",
            fuel_temperature=300.0,
            oxidizer_temperature=500.0
        ),
        operating=OperatingConditions(
            inlet_pressure=45e5,
            pressure_drop=20e5,
            mass_flow_fuel=0.20,
            mass_flow_oxidizer=0.80,
            oxidizer_velocity=100.0
        ),
        geometry=GeometryConfig(
            num_elements=3,
            num_ports=3,
            post_thickness=0.5e-3,
            spray_half_angle=60.0
        )
    )
    
    # Calculate LCSC injector
    print("\n" + "=" * 60)
    print("LCSC Injector Calculation")
    print("=" * 60)
    
    lcsc_calc = LCSCCalculator(config)
    lcsc_results = lcsc_calc.calculate()
    print(lcsc_results)
    
    # Compare discharge coefficients
    print("\n--- Discharge Coefficient Comparison ---")
    cd_comparison = lcsc_calc.compare_discharge_coefficients()
    for name, value in cd_comparison.items():
        print(f"  {name}: {value:.4f}")
    
    # Calculate cold flow equivalent
    print("\n" + "=" * 60)
    print("Cold Flow Equivalent")
    print("=" * 60)
    
    cold_config = ColdFlowConfig(
        inlet_pressure=21e5,
        pressure_drop=20e5
    )
    
    cold_calc = ColdFlowCalculator(
        hot_fire_results=lcsc_results,
        cold_flow_config=cold_config,
        geometry_config=config.geometry
    )
    cold_results = cold_calc.calculate(is_gcsc=False)
    print(cold_results.summary())
    
    # Quick design example
    print("\n" + "=" * 60)
    print("Quick Design Example")
    print("=" * 60)
    
    quick_results = quick_design(
        mass_flow_fuel=0.15,
        mass_flow_oxidizer=0.60,
        pressure_drop=15e5,
        injector_type="LCSC"
    )
    print(f"Quick design orifice diameter: {quick_results.geometry.orifice_diameter:.2f} mm")
    print(f"Quick design momentum flux ratio: {quick_results.performance.momentum_flux_ratio:.2f}")
    
    # Save configuration for later use
    print("\n--- Saving Configuration ---")
    config.to_yaml("example_config.yaml")
    config.to_json("example_config.json")
    print("Configuration saved to example_config.yaml and example_config.json")
