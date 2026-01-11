"""
Phase 3 Feature Demonstration

Shows how to use:
- Operating envelope generation
- Interactive visualizations
- HTML report generation
"""

import sys

from torch_igniter_advanced.analysis import EnvelopeGenerator, find_optimal_mixture_ratio
from torch_igniter_advanced.visualization import plot_mixture_ratio_sweep, plot_pressure_sweep, plot_2d_envelope, \
    plot_geometry_schematic, plot_mass_flow_breakdown, plot_performance_summary

from torch_igniter_advanced import (
    IgniterConfig,
    IgniterDesigner,


)

print("=" * 70)
print("PHASE 3 FEATURES DEMONSTRATION")
print("=" * 70)
print()

# Create baseline configuration
print("Creating baseline configuration...")
config = IgniterConfig(
    name="Phase3_Demo_Igniter",
    description="Demonstration of Phase 3 analysis and reporting capabilities",
    chamber_pressure=20e5,           # 20 bar
    mixture_ratio=2.0,               # O/F = 2.0
    total_mass_flow=0.050,           # 50 g/s
    ethanol_feed_pressure=25e5,      # 25 bar
    n2o_feed_pressure=30e5,          # 30 bar
    ethanol_feed_temperature=298.15, # 25°C
    n2o_feed_temperature=293.15,     # 20°C
    l_star=1.0,
    expansion_ratio=3.0,
    n2o_orifice_count=4,
    ethanol_orifice_count=4,
    discharge_coefficient=0.7
)
print("✓ Configuration created")
print()

# Design the igniter
print("Running igniter design...")
designer = IgniterDesigner()
results = designer.design(config)
print("✓ Design complete")
print()

print(f"Key Results:")
print(f"  Chamber Pressure: {results.chamber_pressure/1e5:.1f} bar")
print(f"  Flame Temperature: {results.flame_temperature:.0f} K")
print(f"  C*: {results.c_star:.1f} m/s")
print(f"  Isp: {results.isp_theoretical:.1f} s")
print(f"  Thrust: {results.thrust:.1f} N")
print(f"  Heat Power: {results.heat_power_output/1000:.1f} kW")
print()

# ============================================================================
# ANALYSIS FEATURES
# ============================================================================

print("=" * 70)
print("ANALYSIS FEATURES")
print("=" * 70)
print()

envelope_gen = EnvelopeGenerator()

# 1. Mixture Ratio Sweep
print("1. Generating mixture ratio sweep (1.5 to 3.0)...")
mr_sweep = envelope_gen.generate_mixture_ratio_sweep(
    config,
    mr_range=(1.5, 3.0),
    n_points=20
)
print(f"✓ Generated {len(mr_sweep)} data points")
print(f"  MR range: {mr_sweep['mixture_ratio'].min():.2f} - {mr_sweep['mixture_ratio'].max():.2f}")
print(f"  Isp range: {mr_sweep['isp'].min():.1f} - {mr_sweep['isp'].max():.1f} s")
print()

# 2. Pressure Sweep
print("2. Generating pressure sweep (10 to 30 bar)...")
p_sweep = envelope_gen.generate_pressure_sweep(
    config,
    pressure_range=(10e5, 30e5),
    n_points=20
)
print(f"✓ Generated {len(p_sweep)} data points")
print(f"  Pressure range: {p_sweep['chamber_pressure'].min():.1f} - {p_sweep['chamber_pressure'].max():.1f} bar")
print(f"  C* range: {p_sweep['c_star'].min():.1f} - {p_sweep['c_star'].max():.1f} m/s")
print()

# 3. 2D Operating Envelope
print("3. Generating 2D operating envelope...")
envelope_2d = envelope_gen.generate_2d_envelope(
    config,
    mr_range=(1.5, 3.0),
    pressure_range=(10e5, 30e5),
    n_mr=10,
    n_p=10
)
print(f"✓ Generated {len(envelope_2d)} data points")
print(f"  Grid: {len(envelope_2d['mixture_ratio'].unique())} MR × {len(envelope_2d['chamber_pressure'].unique())} P")
print()

# 4. Find Optimal Mixture Ratio
print("4. Finding optimal mixture ratio for maximum Isp...")
optimal_mr, opt_sweep = find_optimal_mixture_ratio(
    config,
    mr_range=(1.5, 3.0),
    objective='isp'
)
print(f"✓ Optimal MR: {optimal_mr:.2f}")
max_isp = opt_sweep.loc[opt_sweep['mixture_ratio'] == optimal_mr, 'isp'].values[0]
print(f"  Maximum Isp: {max_isp:.1f} s")
print()

# ============================================================================
# VISUALIZATION FEATURES
# ============================================================================

print("=" * 70)
print("VISUALIZATION FEATURES")
print("=" * 70)
print()

# 1. Mixture Ratio Sweep Plot
print("1. Creating mixture ratio sweep plot...")
fig_mr = plot_mixture_ratio_sweep(mr_sweep)
mr_plot_path = "outputs/mr_sweep.html"
fig_mr.write_html(mr_plot_path)
print(f"✓ Saved to: {mr_plot_path}")
print()

# 2. Pressure Sweep Plot
print("2. Creating pressure sweep plot...")
fig_p = plot_pressure_sweep(p_sweep)
p_plot_path = "outputs/pressure_sweep.html"
fig_p.write_html(p_plot_path)
print(f"✓ Saved to: {p_plot_path}")
print()

# 3. 2D Envelope Contour Plot
print("3. Creating 2D envelope contour plot (Isp)...")
fig_2d_isp = plot_2d_envelope(envelope_2d, parameter='isp')
envelope_isp_path = "outputs/envelope_isp.html"
fig_2d_isp.write_html(envelope_isp_path)
print(f"✓ Saved to: {envelope_isp_path}")
print()

print("4. Creating 2D envelope contour plot (C*)...")
fig_2d_cstar = plot_2d_envelope(envelope_2d, parameter='c_star')
envelope_cstar_path = "outputs/envelope_cstar.html"
fig_2d_cstar.write_html(envelope_cstar_path)
print(f"✓ Saved to: {envelope_cstar_path}")
print()

# 5. Geometry Schematic
print("5. Creating geometry schematic...")
fig_geom = plot_geometry_schematic(results)
geom_path = "outputs/geometry.html"
fig_geom.write_html(geom_path)
print(f"✓ Saved to: {geom_path}")
print()

# 6. Mass Flow Breakdown
print("6. Creating mass flow breakdown chart...")
fig_mass = plot_mass_flow_breakdown(results)
mass_path = "outputs/mass_flow.html"
fig_mass.write_html(mass_path)
print(f"✓ Saved to: {mass_path}")
print()

# 7. Performance Summary
print("7. Creating performance summary chart...")
fig_perf = plot_performance_summary(results)
perf_path = "outputs/performance_summary.html"
fig_perf.write_html(perf_path)
print(f"✓ Saved to: {perf_path}")
print()

# ============================================================================
# REPORT GENERATION
# ============================================================================

print("=" * 70)
print("REPORT GENERATION")
print("=" * 70)
print()

# 1. Quick Report (no envelopes)
print("1. Generating quick report (without operating envelopes)...")
from torch_igniter_advanced.reporting import generate_quick_report, generate_full_report
quick_report_path = "outputs/quick_report.html"
generate_quick_report(config, results, quick_report_path)
print(f"✓ Quick report saved to: {quick_report_path}")
print()

# 2. Full Report (with envelopes)
print("2. Generating full report (with operating envelopes)...")
full_report_path = "outputs/full_report.html"
generate_full_report(config, results, full_report_path)
print(f"✓ Full report saved to: {full_report_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 70)
print("PHASE 3 DEMONSTRATION COMPLETE")
print("=" * 70)
print()

print("Generated Files:")
print("  Analysis Data:")
print(f"    - {len(mr_sweep)} points: Mixture ratio sweep")
print(f"    - {len(p_sweep)} points: Pressure sweep")
print(f"    - {len(envelope_2d)} points: 2D operating envelope")
print()
print("  Visualizations:")
print(f"    - {mr_plot_path}")
print(f"    - {p_plot_path}")
print(f"    - {envelope_isp_path}")
print(f"    - {envelope_cstar_path}")
print(f"    - {geom_path}")
print(f"    - {mass_path}")
print(f"    - {perf_path}")
print()
print("  Reports:")
print(f"    - {quick_report_path}")
print(f"    - {full_report_path}")
print()

print("=" * 70)
print("All Phase 3 features demonstrated successfully!")
print("=" * 70)
