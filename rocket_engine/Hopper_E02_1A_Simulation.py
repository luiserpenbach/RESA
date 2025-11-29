# --- Example Usage ---
from rocket_engine.src.analysis.fluid_state import plot_n2o_p_t_diagram
from rocket_engine.src.engine import EngineConfig, LiquidEngine

if __name__ == "__main__":
    # 1. Define Design Point
    conf = EngineConfig(
        engine_name="HOPPER E1-1A",
        fuel="Ethanol90",
        oxidizer="N2O",
        thrust_n=2200,
        pc_bar=25,
        mr=4.0,
        p_exit_bar=1.013,
        contraction_ratio=12,
        expansion_ratio=0,

        # Geometry
        channel_width_throat=1.0e-3,
        channel_height=0.75e-3,
        rib_width_throat=0.6e-3,
        wall_thickness=0.5e-3,

        # Coolant
        coolant_p_in_bar=98.0
    )

    engine = LiquidEngine(conf)

    # 2. Design Point Run
    res_design = engine.design(plot=True)
    engine.save_specification(output_dir="output-H_E2_1A", tag="nominal")
    #engine.analyze_transient(duration=0.5)

    # 3. Off-Design Runs
    # Case A: Throttled (Low Pc, Low Flow)
    res_throttle = engine.analyze(pc_bar=15.0, mr=3.5, plot=False)

    # 4. Compare in Phase Diagram
    print("\nGenerating Multi-Run Phase Diagram...")

    results_map = {
        "Design (25 bar, MR 4.0)": res_design,
        "Throttled (15 bar, MR 3.5)": res_throttle,
    }

    plot_n2o_p_t_diagram(results_map)
