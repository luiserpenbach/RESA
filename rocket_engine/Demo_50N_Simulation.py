# --- Example Usage ---
from rocket_engine.src.analysis.fluid_state import plot_n2o_p_t_diagram
from rocket_engine.src.engine import EngineConfig, LiquidEngine

if __name__ == "__main__":
    # 1. Define Design Point
    conf = EngineConfig(
        engine_name="Demo-50N",
        fuel="C2H6",
        oxidizer="N2O",
        thrust_n=50,
        pc_bar=7,
        mr=6.0,
        p_exit_bar=1.013,
        contraction_ratio=15,
        expansion_ratio=40,

        # Geometry
        channel_width_throat=0.3e-3,
        channel_height=0.5e-3,
        rib_width_throat=0.5e-3,
        wall_thickness=0.4e-3,

        # Coolant
        coolant_p_in_bar=55.0
    )

    engine = LiquidEngine(conf)

    # 2. Design Point Run
    res_design = engine.design(plot=True)
    engine.save_specification(output_dir="output/output-demo50N", tag="nominal")
    #engine.analyze_transient(duration=0.5)

    # 3. Off-Design Runs
    # Case A: Throttled (Low Pc, Low Flow)
    res_throttle = engine.analyze(pc_bar=4.0, mr=4, plot=False)

    # 4. Compare in Phase Diagram
    print("\nGenerating Multi-Run Phase Diagram...")

    results_map = {
        "Design (25 bar, MR 4.0)": res_design,
        "Throttled (15 bar, MR 3.5)": res_throttle,
    }

    plot_n2o_p_t_diagram(results_map)
