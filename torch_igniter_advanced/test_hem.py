"""
Test HEM injector sizing for N2O and incompressible flow for ethanol.
"""

import sys

from torch_igniter_simple.injector import InjectorDesigner
import numpy as np

print("=" * 70)
print("HEM Injector Sizing Test")
print("=" * 70)
print()

# Test conditions
chamber_pressure = 20e5  # 20 bar
n2o_feed_pressure = 30e5  # 30 bar
ethanol_feed_pressure = 25e5  # 25 bar
n2o_feed_temp = 293.15  # 20°C
ethanol_feed_temp = 298.15  # 25°C

# Mass flows for O/F = 2.0, total 50 g/s
total_mass_flow = 0.050  # kg/s
mixture_ratio = 2.0
fuel_mass_flow = total_mass_flow / (1 + mixture_ratio)
oxidizer_mass_flow = total_mass_flow - fuel_mass_flow

print("Test Conditions:")
print(f"  Chamber Pressure: {chamber_pressure/1e5:.1f} bar")
print(f"  N2O Feed Pressure: {n2o_feed_pressure/1e5:.1f} bar")
print(f"  Ethanol Feed Pressure: {ethanol_feed_pressure/1e5:.1f} bar")
print(f"  N2O Feed Temperature: {n2o_feed_temp:.1f} K ({n2o_feed_temp-273.15:.1f} °C)")
print(f"  Ethanol Feed Temperature: {ethanol_feed_temp:.1f} K ({ethanol_feed_temp-273.15:.1f} °C)")
print()
print(f"  N2O Mass Flow: {oxidizer_mass_flow*1000:.2f} g/s")
print(f"  Ethanol Mass Flow: {fuel_mass_flow*1000:.2f} g/s")
print()

# Initialize designer
designer = InjectorDesigner()

# Test 1: N2O HEM sizing
print("=" * 70)
print("Test 1: N2O Injector (HEM Method)")
print("=" * 70)

try:
    n2o_results = designer.size_n2o_orifice_hem(
        mass_flow=oxidizer_mass_flow,
        feed_pressure=n2o_feed_pressure,
        chamber_pressure=chamber_pressure,
        orifice_count=4,
        discharge_coefficient=0.7,
        feed_temperature=n2o_feed_temp
    )
    
    print("N2O Injector Results:")
    print(f"  Orifice Diameter: {n2o_results['orifice_diameter']*1000:.3f} mm")
    print(f"  Total Area: {n2o_results['total_area']*1e6:.3f} mm²")
    print(f"  Injection Velocity: {n2o_results['injection_velocity']:.1f} m/s")
    print(f"  Pressure Drop: {n2o_results['pressure_drop']/1e5:.1f} bar")
    print(f"  Mass Flux: {n2o_results['mass_flux']:.1f} kg/m²-s")
    print(f"  Exit Quality: {n2o_results['quality_exit']:.3f}")
    print(f"  Flow Choked: {n2o_results['choked']}")
    print()
    
    # Sanity checks
    print("Sanity Checks:")
    calculated_mass_flow = n2o_results['mass_flux'] * n2o_results['total_area'] * 0.7
    print(f"  Mass flow check: {calculated_mass_flow*1000:.2f} g/s (target: {oxidizer_mass_flow*1000:.2f} g/s)")
    
    if n2o_results['orifice_diameter'] > 0 and n2o_results['orifice_diameter'] < 0.01:
        print("  ✓ Orifice diameter reasonable (0-10 mm)")
    else:
        print(f"  ✗ Orifice diameter unusual: {n2o_results['orifice_diameter']*1000:.3f} mm")
    
    if 0 <= n2o_results['quality_exit'] <= 1:
        print("  ✓ Quality in valid range [0, 1]")
    else:
        print(f"  ✗ Quality out of range: {n2o_results['quality_exit']:.3f}")
    
    print()
    
except Exception as e:
    print(f"ERROR in N2O sizing: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Ethanol incompressible sizing
print("=" * 70)
print("Test 2: Ethanol Injector (Incompressible Flow)")
print("=" * 70)

try:
    ethanol_results = designer.size_ethanol_orifice(
        mass_flow=fuel_mass_flow,
        feed_pressure=ethanol_feed_pressure,
        chamber_pressure=chamber_pressure,
        orifice_count=4,
        discharge_coefficient=0.7,
        feed_temperature=ethanol_feed_temp
    )
    
    print("Ethanol Injector Results:")
    print(f"  Orifice Diameter: {ethanol_results['orifice_diameter']*1000:.3f} mm")
    print(f"  Total Area: {ethanol_results['total_area']*1e6:.3f} mm²")
    print(f"  Injection Velocity: {ethanol_results['injection_velocity']:.1f} m/s")
    print(f"  Pressure Drop: {ethanol_results['pressure_drop']/1e5:.1f} bar")
    print(f"  Mass Flux: {ethanol_results['mass_flux']:.1f} kg/m²-s")
    print()
    
    # Sanity checks
    print("Sanity Checks:")
    calculated_mass_flow = ethanol_results['mass_flux'] * ethanol_results['total_area'] * 0.7
    print(f"  Mass flow check: {calculated_mass_flow*1000:.2f} g/s (target: {fuel_mass_flow*1000:.2f} g/s)")
    
    if ethanol_results['orifice_diameter'] > 0 and ethanol_results['orifice_diameter'] < 0.01:
        print("  ✓ Orifice diameter reasonable (0-10 mm)")
    else:
        print(f"  ✗ Orifice diameter unusual: {ethanol_results['orifice_diameter']*1000:.3f} mm")
    
    print()
    
except Exception as e:
    print(f"ERROR in ethanol sizing: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Combined sizing
print("=" * 70)
print("Test 3: Combined Injector Sizing")
print("=" * 70)

try:
    all_results = designer.size_all_injectors(
        n2o_mass_flow=oxidizer_mass_flow,
        ethanol_mass_flow=fuel_mass_flow,
        n2o_feed_pressure=n2o_feed_pressure,
        ethanol_feed_pressure=ethanol_feed_pressure,
        chamber_pressure=chamber_pressure,
        n2o_orifice_count=4,
        ethanol_orifice_count=4,
        discharge_coefficient=0.7,
        n2o_feed_temperature=n2o_feed_temp,
        ethanol_feed_temperature=ethanol_feed_temp
    )
    
    print("Complete Injector Design:")
    print()
    print("N2O:")
    print(f"  Diameter: {all_results['n2o']['orifice_diameter']*1000:.3f} mm")
    print(f"  Velocity: {all_results['n2o']['injection_velocity']:.1f} m/s")
    print()
    print("Ethanol:")
    print(f"  Diameter: {all_results['ethanol']['orifice_diameter']*1000:.3f} mm")
    print(f"  Velocity: {all_results['ethanol']['injection_velocity']:.1f} m/s")
    print()
    
    # Comparison
    diameter_ratio = all_results['n2o']['orifice_diameter'] / all_results['ethanol']['orifice_diameter']
    print(f"N2O/Ethanol Diameter Ratio: {diameter_ratio:.2f}")
    print(f"(N2O orifices typically larger due to two-phase flow)")
    
    print()
    print("=" * 70)
    print("HEM Implementation Complete!")
    print("=" * 70)
    
except Exception as e:
    print(f"ERROR in combined sizing: {e}")
    import traceback
    traceback.print_exc()
