# --- Example Usage ---
import numpy as np

from rocket_engine.src.analysis.fluid_state import plot_n2o_p_t_diagram, plot_n2o_t_rho_diagram
from rocket_engine.src.analysis.performance import plot_cstar_contour, plot_isp_vs_mr
from rocket_engine.src.engine import EngineConfig, LiquidEngine
from rocket_engine.src.visualization import plot_channel_cross_section_radial
from rocket_engine.src.visualization_3d import plot_coolant_channels_3d

if __name__ == "__main__":
    # 1. Define Design Point TODO: get from YAML config file
    conf = EngineConfig(
        engine_name="HOPPER E1-1A",
        fuel="Ethanol90",
        oxidizer="N2O",
        thrust_n=2200.0,
        pc_bar=25.0,
        mr=4.0,
        p_exit_bar=1.013,

        # Chamber geometry

        L_star=1100.0,
        contraction_ratio=12.0,
        expansion_ratio=4.1,
        theta_convergent=35.0,

        # Cooling Geometry
        channel_width_throat=1.0e-3,
        channel_height=0.75e-3,
        rib_width_throat=0.6e-3,
        wall_thickness=0.5e-3,
        wall_roughness=20e-6,

        # Coolant State
        coolant_name="REFPROP::NitrousOxide",
        coolant_p_in_bar=97.0,
        coolant_t_in_k=298.0,       # 25°C
        cooling_mode='counter-flow'
    )

    data_output_dir = "output/output_data_H-E2-1A-V1.0"
    plots_output_dir = "output/output_plots_H-E2-1A-V1.0"

    engine = LiquidEngine(conf)


    # 2. RUN DESIGN POINT
    # -------------------
    print("\n--- [1] Running Design Point Sizing ---")
    res_design = engine.design(plot=False)
    engine.save_specification(output_dir=data_output_dir, tag="design")

    print(f"   >> Design Thrust: {res_design.thrust_sea:.1f} N (Sea-Level)")
    print(f"   >> SL Isp:    {res_design.isp_sea:.1f} s")
    print(f"   >> Max Wall Temp: {np.max(res_design.cooling_data['T_wall_hot']):.1f} K")


    # 3. RUN OFF-DESIGN ANALYSIS
    # --------------------------
    print("\n--- [2] Running Off-Design Simulations ---")
    #engine.analyze_transient(duration=0.5)

    # Case A: Oxidizer Throttled (Passive Fuel)
    # Simulate Oxidizer Valve setting with a fixed Fuel Venturi
    res_throttle_75 = engine.analyze_oxidizer_throttle(
        ox_flow_fraction=0.75,
        fixed_fuel_flow=True
    )
    res_throttle_50 = engine.analyze_oxidizer_throttle(
        ox_flow_fraction=0.50,
        fixed_fuel_flow=True
    )
    engine.save_specification(data_output_dir, tag="throttle_70pct")

    # Case B: Manual Off-Design Hot Run
    res_hot = engine.analyze(pc_bar=30.0, mr=5.0, plot=False)
    engine.save_specification(data_output_dir, tag="hot_case")




    # 4. GENERATE VISUALIZATIONS
    # -----------------------------------
    print("\n--- [3] Generating Visualizations ---")

    results_map = {
        "Design (25 bar, MR 4.0)": res_design,
        "Throttled (80% Ox) ": res_throttle_75,
        "Throttled (50% Ox) ": res_throttle_50,
    }
    #plot_n2o_p_t_diagram(results_map)

    # B. Detailed T-Rho Diagram for Design Point
    print("   -> Plotting T-Rho State Path...")
    #plot_n2o_t_rho_diagram(res_design)

    # C. Throat Cross-Section
    '''
    print("   -> Plotting Throat Cross-Section...")
    idx_throat = np.argmin(res_design.geometry.y_full)
    plot_channel_cross_section_radial(
        res_design.channel_geometry,
        station_idx=idx_throat,
        sector_angle=90
    )
    '''

    # D. 3D Channel View
    #print("   -> Generating 3D Channel Visualization...")
    #plot_coolant_channels_3d(res_design.channel_geometry, num_channels_to_show=5)

    # E1. c* Performance Map
    plot_cstar_contour(
        engine.cea,
        nominal_pc_bar=conf.pc_bar,
        nominal_mr=conf.mr,
        pc_range=(10, 60),
        mr_range=(1.5, 7.0)
    )

    # E2. c* Performance Map with Ox Throttle Path Data
    # Define Operating Limits for Plot
    op_envelope = {
        'pc_min': 12.0,
        'pc_max': 30.0,
        'mr_min': 1.9,
        'mr_max': 4.5
    }
    plot_cstar_contour(
        engine.cea,
        nominal_pc_bar=conf.pc_bar,
        nominal_mr=conf.mr,
        mr_range=(1.5, 7.0),
        pc_range=(5.0, 60.0),  # Ensure range covers the throttled Pc
        trajectory=engine.generate_throttle_curve(start_pct=100, end_pct=50, fixed_fuel=True),
        envelope = op_envelope
    )
    # F. Isp Sensitivity Curves
    plot_isp_vs_mr(
        engine.cea,
        nominal_pc_bar=conf.pc_bar,
        nominal_mr=conf.mr,
        pc_levels=[15, 25, 40],
        expansion_ratio=res_design.expansion_ratio
    )

    print("\n✅ All tasks completed successfully.")
