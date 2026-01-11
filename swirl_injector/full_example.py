"""
Comprehensive example script demonstrating the refactored swirl injector toolkit.

This script shows:
1. Loading configuration from files
2. Running LCSC and GCSC calculations
3. Cold flow equivalent sizing
4. Visualization and sensitivity analysis (if plotly available)
5. Saving results
"""
import sys
sys.path.insert(0, 'C:/Users/erpen/Documents/GitHub-local/RESA/swirl_injector')

import numpy as np
from pathlib import Path

from config import InjectorConfig, ColdFlowConfig, PropellantConfig, OperatingConditions, GeometryConfig
from calculators import LCSCCalculator, GCSCCalculator, ColdFlowCalculator

# Try to import visualization (optional)
try:
    from visualization import (
        plot_injector_cross_section,
        plot_discharge_coefficient_comparison,
        plot_spray_angle_comparison,
        plot_sensitivity_analysis,
        plot_operating_envelope,
        save_figure,
        PLOTLY_AVAILABLE
    )
except ImportError:
    PLOTLY_AVAILABLE = False


def run_lcsc_example():
    """Run LCSC injector design example."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: LCSC Injector Design")
    print("=" * 70)
    
    # Create configuration
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
    
    # Run calculation
    calc = LCSCCalculator(config)
    results = calc.calculate()
    
    # Print results
    print(results)
    
    # Compare discharge coefficients
    print("\n--- Discharge Coefficient Comparison ---")
    cd_comp = calc.compare_discharge_coefficients()
    for name, value in cd_comp.items():
        print(f"  {name:20s}: {value:.4f}")
    
    return results, config


def run_gcsc_example():
    """Run GCSC injector design example."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: GCSC Injector Design")
    print("=" * 70)
    
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
            oxidizer_velocity=150.0  # Higher velocity for GCSC
        ),
        geometry=GeometryConfig(
            num_elements=3,
            num_ports=3,
            post_thickness=0.5e-3,
            minimum_clearance=0.5e-3
        )
    )
    
    calc = GCSCCalculator(config)
    results = calc.calculate()
    
    print(results)
    
    return results, config


def run_cold_flow_example(hot_fire_results, geometry_config):
    """Run cold flow equivalent calculation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Cold Flow Equivalent Sizing")
    print("=" * 70)
    
    cold_config = ColdFlowConfig(
        inlet_pressure=21e5,
        pressure_drop=20e5,
        ambient_temperature=293.15,
        gas_fluid="nitrogen",
        liquid_fluid="water"
    )
    
    calc = ColdFlowCalculator(
        hot_fire_results=hot_fire_results,
        cold_flow_config=cold_config,
        geometry_config=geometry_config
    )
    
    results = calc.calculate(is_gcsc=False)
    print(results.summary())
    
    return results


def run_parametric_study(config):
    """Run parametric study on key parameters."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Parametric Study")
    print("=" * 70)
    
    # Study effect of spray half angle
    angles = np.linspace(40, 80, 9)
    print("\nEffect of Spray Half Angle:")
    print("-" * 60)
    print(f"{'Angle (°)':>10} {'J':>10} {'We':>10} {'r_o (mm)':>12} {'C_D':>10}")
    print("-" * 60)
    
    for angle in angles:
        config_dict = config.to_dict()
        config_dict['geometry']['spray_half_angle'] = angle
        mod_config = InjectorConfig.from_dict(config_dict)
        
        calc = LCSCCalculator(mod_config)
        result = calc.calculate()
        
        print(f"{angle:>10.1f} {result.performance.momentum_flux_ratio:>10.3f} "
              f"{result.performance.weber_number:>10.1f} "
              f"{result.geometry.orifice_radius * 1000:>12.3f} "
              f"{result.performance.discharge_coefficient:>10.4f}")
    
    # Study effect of pressure drop
    pressure_drops = np.linspace(10e5, 30e5, 5)
    print("\nEffect of Pressure Drop:")
    print("-" * 60)
    print(f"{'ΔP (bar)':>10} {'J':>10} {'We':>10} {'r_o (mm)':>12} {'v_f (m/s)':>12}")
    print("-" * 60)
    
    for dp in pressure_drops:
        config_dict = config.to_dict()
        config_dict['operating']['pressure_drop'] = dp
        mod_config = InjectorConfig.from_dict(config_dict)
        
        calc = LCSCCalculator(mod_config)
        result = calc.calculate()
        
        # Estimate axial velocity from film thickness
        t_film = result.performance.film_thickness
        r_o = result.geometry.orifice_radius
        r_aircore = r_o - t_film
        rho_f = result.propellant_properties.fuel_at_chamber.density
        v_f = config.operating.mass_flow_fuel / (
            config.geometry.num_elements * rho_f * np.pi * (r_o**2 - r_aircore**2)
        )
        
        print(f"{dp/1e5:>10.1f} {result.performance.momentum_flux_ratio:>10.3f} "
              f"{result.performance.weber_number:>10.1f} "
              f"{result.geometry.orifice_radius * 1000:>12.3f} "
              f"{v_f:>12.2f}")


def run_design_comparison():
    """Compare different design configurations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Design Comparison (LCSC vs GCSC)")
    print("=" * 70)
    
    # Base configuration
    base_config = InjectorConfig(
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
    
    # Calculate both types
    lcsc_calc = LCSCCalculator(base_config)
    lcsc_results = lcsc_calc.calculate()
    
    gcsc_calc = GCSCCalculator(base_config)
    gcsc_results = gcsc_calc.calculate()
    
    # Comparison table
    print("\n" + "-" * 60)
    print(f"{'Parameter':30s} {'LCSC':>12} {'GCSC':>12}")
    print("-" * 60)
    
    comparisons = [
        ("Orifice radius (mm)", 
         lcsc_results.geometry.orifice_radius * 1000,
         gcsc_results.geometry.orifice_radius * 1000),
        ("Port radius (mm)",
         lcsc_results.geometry.port_radius * 1000,
         gcsc_results.geometry.port_radius * 1000),
        ("Swirl chamber radius (mm)",
         lcsc_results.geometry.swirl_chamber_radius * 1000,
         gcsc_results.geometry.swirl_chamber_radius * 1000),
        ("Spray half angle (°)",
         lcsc_results.performance.spray_half_angle,
         gcsc_results.performance.spray_half_angle),
        ("Swirl number",
         lcsc_results.performance.swirl_number,
         gcsc_results.performance.swirl_number),
        ("Momentum flux ratio (J)",
         lcsc_results.performance.momentum_flux_ratio,
         gcsc_results.performance.momentum_flux_ratio),
        ("Weber number",
         lcsc_results.performance.weber_number,
         gcsc_results.performance.weber_number),
        ("Discharge coefficient",
         lcsc_results.performance.discharge_coefficient,
         gcsc_results.performance.discharge_coefficient),
        ("Film thickness (mm)",
         lcsc_results.performance.film_thickness * 1000,
         gcsc_results.performance.film_thickness * 1000),
    ]
    
    for name, lcsc_val, gcsc_val in comparisons:
        print(f"{name:30s} {lcsc_val:>12.3f} {gcsc_val:>12.3f}")


def generate_visualizations(results, config, output_dir: Path):
    """Generate and save visualization plots."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Generating Visualizations")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cross-section plot
    print("  - Generating injector cross-section...")
    fig1 = plot_injector_cross_section(results, show=False)
    save_figure(fig1, output_dir / "injector_cross_section.html")
    
    # Discharge coefficient comparison
    print("  - Generating discharge coefficient comparison...")
    fig2 = plot_discharge_coefficient_comparison(show=False)
    save_figure(fig2, output_dir / "cd_comparison.html")
    
    # Spray angle comparison
    print("  - Generating spray angle comparison...")
    fig3 = plot_spray_angle_comparison(show=False)
    save_figure(fig3, output_dir / "spray_angle_comparison.html")
    
    print(f"\nVisualization files saved to: {output_dir}")


def main():
    """Run all examples."""
    print("=" * 70)
    print("SWIRL INJECTOR DIMENSIONING TOOLKIT - EXAMPLES")
    print("=" * 70)
    
    # Example 1: LCSC Design
    lcsc_results, lcsc_config = run_lcsc_example()
    
    # Example 2: GCSC Design
    gcsc_results, gcsc_config = run_gcsc_example()
    
    # Example 3: Cold Flow Equivalent
    cold_results = run_cold_flow_example(lcsc_results, lcsc_config.geometry)
    
    # Example 4: Parametric Study
    run_parametric_study(lcsc_config)
    
    # Example 5: Design Comparison
    run_design_comparison()
    
    # Example 6: Visualizations (if plotly available)
    output_dir = Path("C:/Users/erpen/Documents/GitHub-local/RESA/swirl_injector/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if PLOTLY_AVAILABLE:
        generate_visualizations(lcsc_results, lcsc_config, output_dir)
    else:
        print("\n" + "=" * 70)
        print("SKIPPING VISUALIZATIONS (Plotly not available)")
        print("Install with: pip install plotly")
        print("=" * 70)
    
    # Save results
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)
    
    lcsc_results.to_json(output_dir / "lcsc_results.json")
    lcsc_config.to_yaml(output_dir / "lcsc_config.yaml")
    print(f"Results saved to: {output_dir}")
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
