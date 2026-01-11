# --- Example Usage ---
from rocket_engine.src.analysis.fluid_state import plot_n2o_p_t_diagram
from rocket_engine.src.engine import EngineConfig, LiquidEngine

if __name__ == "__main__":
    # 1. Define Design Point
    conf = EngineConfig(
        engine_name="Demo 45KN",
        fuel="CH4",
        oxidizer="LOX",
        thrust_n=40_000,
        pc_bar=60,
        mr=3.0,
        p_exit_bar=1,
        contraction_ratio=5,
        expansion_ratio=0,
        L_star=1000,

        # Geometry
        channel_width_throat=1.5e-3,
        channel_height=3e-3,
        rib_width_throat=1e-3,
        wall_thickness=0.9e-3,

        # Coolant
        coolant_name="REFPROP::Methane",
        coolant_p_in_bar=100.0,
        coolant_t_in_k=100
    )

    engine = LiquidEngine(conf)

    # 2. Design Point Run
    res_design = engine.design(plot=True)
    engine.save_specification("demo_40KN")

    # 3. Off-Design Runs
    # Case A: Throttled (Low Pc, Low Flow)
    res_throttle = engine.analyze(pc_bar=40.0, mr=2.8, plot=False)
    # Case B: Hot Run (High MR, slightly higher Pc)
    res_hot = engine.analyze(pc_bar=80.0, mr=3.0, plot=False)

    # 4. Compare in Phase Diagram
    print("\nGenerating Multi-Run Phase Diagram...")

    results_map = {
        "Design (60 bar, MR 3.0)": res_design,
        "Throttled (40 bar, MR 2.8)": res_throttle,
        "Hot Run (80 bar, MR 3.0)": res_hot
    }

    plot_n2o_p_t_diagram(results_map)
