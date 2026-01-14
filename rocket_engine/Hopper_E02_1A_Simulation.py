"""
Comprehensive Demo Simulation: HOPPER E1-1A Engine (Ethanol90/N2O)
Updated to use the new resa package (v2.0)

This script demonstrates:
- Design point analysis
- Off-design and throttle analysis
- Comprehensive visualization
- HTML report generation
- Throttle curve generation
"""
import numpy as np
from resa import Engine, EngineConfig
from resa.visualization import (
    EngineDashboardPlotter,
    PerformanceContourPlotter,
    ThrottleCurvePlotter
)
from resa.reporting import HTMLReportGenerator

if __name__ == "__main__":
    # 1. DEFINE DESIGN POINT CONFIGURATION
    print("\n" + "="*70)
    print("HOPPER E1-1A Engine - Comprehensive Design & Analysis")
    print("="*70)

    config = EngineConfig(
        engine_name="HOPPER E1-1A",
        fuel="Ethanol90",
        oxidizer="N2O",
        thrust_n=2200.0,
        pc_bar=25.0,
        mr=4.0,
        p_exit_bar=1.013,
        eff_combustion=0.94,

        # Chamber geometry
        throat_diameter=28.06e-3,
        L_star=1200.0,
        contraction_ratio=12.0,
        expansion_ratio=4.1,
        theta_convergent=35.0,

        # Cooling Geometry
        channel_width_throat=1.0e-3,
        channel_height=0.75e-3,
        rib_width_throat=0.6e-3,
        wall_thickness=0.5e-3,
        wall_roughness=100e-6,

        # Coolant State (N2O regenerative cooling)
        coolant_name="REFPROP::NitrousOxide",
        coolant_p_in_bar=97.0,
        coolant_t_in_k=298.0,  # 25°C
        cooling_mode='counter-flow'
    )

    data_output_dir = "output/output_data_H-E2-1A-V1.0"
    plots_output_dir = "output/output_plots_H-E2-1A-V1.0"

    engine = Engine(config, output_dir=data_output_dir)

    # 2. RUN DESIGN POINT
    print("\n" + "-"*70)
    print("[1] Running Design Point Sizing")
    print("-"*70)

    res_design = engine.design()

    print(f"\n✓ Design Point Results:")
    print(f"   Thrust (vac):        {res_design.thrust_vac:.1f} N")
    print(f"   Thrust (sea level):  {res_design.thrust_sea:.1f} N")
    print(f"   Isp (vac):           {res_design.isp_vac:.1f} s")
    print(f"   Isp (sea level):     {res_design.isp_sea:.1f} s")
    print(f"   Chamber Pressure:    {config.pc_bar:.1f} bar")
    print(f"   Mixture Ratio:       {config.mr:.2f}")
    print(f"   Mass Flow (total):   {res_design.total_mass_flow:.4f} kg/s")
    print(f"   Combustion Temp:     {res_design.combustion.T_combustion:.0f} K")

    if res_design.cooling:
        max_wall_temp = max(res_design.cooling.T_wall_hot)
        print(f"   Max Wall Temp:       {max_wall_temp:.1f} K")
        print(f"   Coolant Outlet Temp: {res_design.cooling.coolant_outlet.T:.1f} K")

    # Generate HTML report for design point
    print("\n   Generating design point HTML report...")
    reporter = HTMLReportGenerator()
    report_path = f"{data_output_dir}/hopper_e1_1a_design_report.html"
    reporter.generate(res_design, output_path=report_path)
    print(f"   Report saved to: {report_path}")

    # 3. RUN OFF-DESIGN ANALYSIS
    print("\n" + "-"*70)
    print("[2] Running Off-Design Simulations")
    print("-"*70)

    # Generate throttle sweep
    print("\n   Generating throttle sweep (50% to 100%)...")
    throttle_curve = engine.throttle_sweep(
        pc_range=(12.5, 25.0),  # 50% to 100% thrust
        mr_range=(3.0, 4.5),
        num_points=10
    )

    print(f"   ✓ Throttle curve generated: {len(throttle_curve.points)} points")

    # Analyze specific throttle points
    print("\n   Analyzing specific off-design conditions:")

    # Case A: 75% Throttle
    print("   - 75% throttle (18.75 bar, MR 3.5)...")
    res_throttle_75 = engine.analyze(pc_bar=18.75, mr=3.5)
    print(f"     Thrust: {res_throttle_75.thrust_vac:.0f} N ({res_throttle_75.thrust_vac/res_design.thrust_vac*100:.0f}%)")

    # Case B: 50% Throttle
    print("   - 50% throttle (12.5 bar, MR 3.0)...")
    res_throttle_50 = engine.analyze(pc_bar=12.5, mr=3.0)
    print(f"     Thrust: {res_throttle_50.thrust_vac:.0f} N ({res_throttle_50.thrust_vac/res_design.thrust_vac*100:.0f}%)")

    # Case C: Hot Run (higher Pc)
    print("   - Hot case (30 bar, MR 5.0)...")
    res_hot = engine.analyze(pc_bar=30.0, mr=5.0)
    print(f"     Thrust: {res_hot.thrust_vac:.0f} N ({res_hot.thrust_vac/res_design.thrust_vac*100:.0f}%)")

    # 4. GENERATE VISUALIZATIONS
    print("\n" + "-"*70)
    print("[3] Generating Visualizations")
    print("-"*70)

    try:
        # A. Engine Dashboard for Design Point
        print("   - Creating engine dashboard...")
        dashboard_plotter = EngineDashboardPlotter()
        fig_dashboard = dashboard_plotter.create_figure(res_design)
        dashboard_path = f"{plots_output_dir}/hopper_e1_1a_dashboard.html"
        fig_dashboard.write_html(dashboard_path)
        print(f"     Saved to: {dashboard_path}")

        # B. Throttle Curve Visualization
        print("   - Creating throttle curve plot...")
        throttle_plotter = ThrottleCurvePlotter()

        # Prepare throttle data in the format expected by the plotter
        throttle_data = []
        for point in throttle_curve.points:
            throttle_pct = (point.thrust / res_design.thrust_vac) * 100
            throttle_data.append({
                'pct': throttle_pct,
                'pc': point.pc_bar,
                'mr': point.mr,
                'thrust': point.thrust,
                'isp_vac': point.isp_vac
            })

        fig_throttle = throttle_plotter.create_figure(
            throttle_data,
            design_point={
                'pct': 100,
                'pc': config.pc_bar,
                'mr': config.mr,
                'thrust': res_design.thrust_vac
            }
        )
        throttle_path = f"{plots_output_dir}/hopper_e1_1a_throttle_curve.html"
        fig_throttle.write_html(throttle_path)
        print(f"     Saved to: {throttle_path}")

        # C. Performance Contours
        print("   - Creating performance contour plots...")
        perf_plotter = PerformanceContourPlotter()

        # Prepare data for contour plot
        pc_range = np.linspace(10, 40, 15)
        mr_range = np.linspace(2.0, 6.0, 15)

        # Create meshgrid for contour
        PC, MR = np.meshgrid(pc_range, mr_range)
        isp_grid = np.zeros_like(PC)

        print("     Calculating performance map...")
        for i in range(len(mr_range)):
            for j in range(len(pc_range)):
                try:
                    result = engine.analyze(pc_bar=PC[i,j], mr=MR[i,j])
                    isp_grid[i,j] = result.isp_vac
                except Exception:
                    isp_grid[i,j] = np.nan

        # Create contour figure using Plotly
        import plotly.graph_objects as go
        fig_contour = go.Figure()

        fig_contour.add_trace(go.Contour(
            x=pc_range,
            y=mr_range,
            z=isp_grid,
            colorscale='Viridis',
            contours=dict(
                start=200,
                end=280,
                size=5
            ),
            colorbar=dict(title="Isp (s)")
        ))

        # Add design point
        fig_contour.add_trace(go.Scatter(
            x=[config.pc_bar],
            y=[config.mr],
            mode='markers',
            marker=dict(size=12, color='red', symbol='star'),
            name='Design Point'
        ))

        # Add throttle trajectory
        throttle_pc = [point.pc_bar for point in throttle_curve.points]
        throttle_mr = [point.mr for point in throttle_curve.points]
        fig_contour.add_trace(go.Scatter(
            x=throttle_pc,
            y=throttle_mr,
            mode='lines+markers',
            line=dict(color='white', width=2, dash='dash'),
            marker=dict(size=6, color='white'),
            name='Throttle Path'
        ))

        fig_contour.update_layout(
            title="Isp Performance Map - HOPPER E1-1A",
            xaxis_title="Chamber Pressure (bar)",
            yaxis_title="Mixture Ratio (O/F)",
            width=900,
            height=700
        )

        contour_path = f"{plots_output_dir}/hopper_e1_1a_performance_contour.html"
        fig_contour.write_html(contour_path)
        print(f"     Saved to: {contour_path}")

    except Exception as e:
        print(f"   ⚠ Visualization error: {e}")
        import traceback
        traceback.print_exc()

    # 5. EXPORT DATA SUMMARY
    print("\n" + "-"*70)
    print("[4] Exporting Data Summary")
    print("-"*70)

    # Export throttle table to CSV
    try:
        import pandas as pd
        throttle_df = pd.DataFrame([
            {
                'Throttle (%)': point.thrust / res_design.thrust_vac * 100,
                'Pc (bar)': point.pc_bar,
                'MR': point.mr,
                'Thrust (N)': point.thrust,
                'Isp (s)': point.isp_vac
            }
            for point in throttle_curve.points
        ])

        csv_path = f"{data_output_dir}/throttle_profile.csv"
        throttle_df.to_csv(csv_path, index=False)
        print(f"   ✓ Throttle data exported to: {csv_path}")
    except Exception as e:
        print(f"   ⚠ CSV export error: {e}")

    # Print summary table
    print("\n" + "-"*70)
    print("RESULTS SUMMARY")
    print("-"*70)
    print(f"\n{'Condition':<20} {'Thrust (N)':<12} {'Isp (s)':<10} {'Pc (bar)':<10} {'MR':<8}")
    print("-"*70)
    print(f"{'Design Point':<20} {res_design.thrust_vac:<12.0f} {res_design.isp_vac:<10.1f} {config.pc_bar:<10.1f} {config.mr:<8.2f}")
    print(f"{'75% Throttle':<20} {res_throttle_75.thrust_vac:<12.0f} {res_throttle_75.isp_vac:<10.1f} {18.75:<10.1f} {3.5:<8.2f}")
    print(f"{'50% Throttle':<20} {res_throttle_50.thrust_vac:<12.0f} {res_throttle_50.isp_vac:<10.1f} {12.5:<10.1f} {3.0:<8.2f}")
    print(f"{'Hot Case':<20} {res_hot.thrust_vac:<12.0f} {res_hot.isp_vac:<10.1f} {30.0:<10.1f} {5.0:<8.2f}")

    print("\n" + "="*70)
    print("✅ All tasks completed successfully.")
    print("="*70)
    print(f"\nOutputs saved to:")
    print(f"  - Data:  {data_output_dir}/")
    print(f"  - Plots: {plots_output_dir}/")
    print()
