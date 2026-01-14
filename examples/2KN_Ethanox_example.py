"""
RESA Example: 2kN Ethanol/N2O Engine
Updated to use the new resa package (v2.0)
"""
from resa import Engine, EngineConfig

# Configure the engine
config = EngineConfig(
    engine_name="Hopper E2-C01",
    fuel="Ethanol90",
    oxidizer="N2O",
    thrust_n=2200,
    pc_bar=25,
    mr=4.0,  # Mixture ratio (O/F)
    L_star=1300,  # L* in mm (1.3m converted)
    contraction_ratio=10,
    expansion_ratio=None,  # Will be calculated for optimal expansion
    nozzle_length_pct=0.8,
)

# Create engine and run design
engine = Engine(config)
result = engine.design()

# Display summary
print("\n" + "="*60)
print(f" {config.engine_name} Design Summary")
print("="*60)
print(f"\nThrust (vacuum):      {result.thrust_vac:.0f} N")
print(f"Isp (vacuum):         {result.isp_vac:.1f} s")
print(f"Chamber Pressure:     {config.pc_bar:.1f} bar")
print(f"Mixture Ratio:        {config.mr:.2f}")
print(f"Mass Flow (total):    {result.total_mass_flow:.4f} kg/s")
print(f"\nThroat Diameter:      {result.nozzle.throat_diameter*1000:.2f} mm")
print(f"Exit Diameter:        {result.nozzle.exit_diameter*1000:.2f} mm")
print(f"Expansion Ratio:      {result.nozzle.expansion_ratio:.2f}")
print(f"\nCombustion Temp:      {result.combustion.T_combustion:.0f} K")
print(f"C* Efficiency:        {result.combustion.cstar_efficiency*100:.1f}%")
print("="*60 + "\n")

# Generate and save HTML report
print("Generating HTML report...")
result.to_html("output/hopper_e2_c01_report.html")
print("Report saved to: output/hopper_e2_c01_report.html")

# Optionally display interactive plots
try:
    from resa import EngineDashboardPlotter
    plotter = EngineDashboardPlotter()
    fig = plotter.create_figure(result)
    fig.show()
except ImportError:
    print("Install plotly for interactive visualizations")
