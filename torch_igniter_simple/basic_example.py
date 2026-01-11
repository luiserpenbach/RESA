"""
Example usage of torch igniter sizing tool.

This script demonstrates basic usage and validates the core functionality.
"""

import sys
#sys.path.insert(0, '/home/claude')

from torch_igniter_simple import IgniterDesigner, IgniterConfig


def create_baseline_config() -> IgniterConfig:
    """Create a baseline igniter configuration."""
    
    config = IgniterConfig(
        # Operating conditions
        chamber_pressure=40e5,      # bar
        mixture_ratio=2.0,          # O/F = 2.0
        total_mass_flow=0.050,      # 50 g/s
        
        # Feed system
        ethanol_feed_pressure=95e5,    # bar
        n2o_feed_pressure=95e5,        # bar
        ethanol_feed_temperature=298.15,  # 25°C
        n2o_feed_temperature=293.15,   # 20°C (cooler N2O)
        
        # Design parameters
        l_star=0.8,                 # 1.0 m characteristic length
        expansion_ratio=2.0,        # Area ratio 3:1
        nozzle_type="conical",
        conical_half_angle=15.0,    # 15 degree half angle
        
        # Injector
        n2o_orifice_count=1,
        ethanol_orifice_count=1,
        discharge_coefficient=0.8,
        
        # Environment
        ambient_pressure=101325.0,  # Sea level
        
        # Metadata
        name="baseline_igniter",
        description="Baseline Ethanol/N2O torch igniter for preliminary sizing"
    )
    
    return config


def main():
    """Run baseline igniter design example."""
    
    print("=" * 70)
    print("TORCH IGNITER SIZING TOOL - Example")
    print("=" * 70)
    print()
    
    # Create configuration
    print("Creating baseline configuration...")
    config = create_baseline_config()

    # Save configuration
    config.save_json("baseline_igniter_config.json")
    print(f"Configuration saved to: baseline_igniter_config.json")
    print()
    
    # Design igniter
    print("Designing igniter...")
    print("(This will call NASA CEA and CoolProp)")
    print()
    
    try:
        designer = IgniterDesigner()
        results = designer.design(config)
        
        # Display results
        print(results.summary())
        print()
        
        # Save results
        results.save_json("outputs/baseline_igniter_results.json")
        print(f"Results saved to: baseline_igniter_results.json")
        print()
        
        # Highlight key metrics
        print("=" * 70)
        print("KEY PERFORMANCE INDICATORS:")
        print("=" * 70)
        print(f"  Heat Power Output:    {results.heat_power_output/1000:.2f} kW")
        print(f"  Thrust:               {results.thrust:.2f} N")
        print(f"  C* Efficiency:        {results.c_star_efficiency*100:.1f}%")
        print(f"  Throat Diameter:      {results.throat_diameter*1000:.2f} mm")
        print(f"  Chamber Volume:       {results.chamber_volume*1e6:.2f} cm³")
        print("=" * 70)
        
        return results
        
    except Exception as e:
        print(f"ERROR: Design failed: {e}")
        print()
        print("This may be due to missing dependencies:")
        print("  - RocketCEA: pip install rocketcea")
        print("  - CoolProp: pip install CoolProp")
        raise


if __name__ == "__main__":
    results = main()
