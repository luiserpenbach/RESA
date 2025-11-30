from rocket_engine.src.analysis.performance import plot_cstar_contour, plot_isp_vs_mr
from rocket_engine.src.physics.combustion import CEASolver

# 1. Initialize Solver
# (Use the same propellants as your engine config)
cea = CEASolver(fuel_name="Ethanol90", oxidizer_name="N2O")

# 2. Define Operating Point
nom_pc = 25.0  # bar
nom_mr = 4.0

# 3. Generate C* Contour Map
# Shows how c* changes with pressure and MR.
# Useful for seeing if you are near the peak c* (max combustion efficiency potential).
plot_cstar_contour(
    cea_solver=cea,
    nominal_pc_bar=nom_pc,
    nominal_mr=nom_mr,
    mr_range=(2.0, 7.0),
    pc_range=(10.0, 60.0)
)

# 4. Generate Isp Curves
# Shows how sensitive performance is to MR shifts (e.g. if a valve is stuck).
plot_isp_vs_mr(
    cea_solver=cea,
    nominal_pc_bar=nom_pc,
    nominal_mr=nom_mr,
    pc_levels=[15, 25, 40, 60], # Compare different pressures
    expansion_ratio=4.0        # Assumed nozzle size
)