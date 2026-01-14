"""
Demo Simulation: 45kN Engine (CH4/LOX)
Updated to use the new resa package (v2.0)
"""
from resa import Engine, EngineConfig
from resa.visualization import EngineDashboardPlotter
from resa.reporting import HTMLReportGenerator

if __name__ == "__main__":
    # 1. Define Design Point Configuration
    config = EngineConfig(
        engine_name="Demo 45KN",
        fuel="CH4",
        oxidizer="LOX",
        thrust_n=40_000,
        pc_bar=60,
        mr=3.0,
        p_exit_bar=1.0,
        contraction_ratio=5,
        expansion_ratio=0,  # Will be calculated for optimal expansion
        L_star=1000,

        # Cooling Geometry
        channel_width_throat=1.5e-3,
        channel_height=3e-3,
        rib_width_throat=1e-3,
        wall_thickness=0.9e-3,

        # Coolant (use fuel as coolant)
        coolant_name="REFPROP::Methane",
        coolant_p_in_bar=100.0,
        coolant_t_in_k=100
    )

    engine = Engine(config, output_dir="output/demo_40KN")

    # 2. Design Point Run
    print("\n" + "="*60)
    print("DESIGN POINT ANALYSIS - 45kN CH4/LOX Engine")
    print("="*60)
    res_design = engine.design()

    print(f"\nDesign Results:")
    print(f"  Thrust (vac):        {res_design.thrust_vac/1000:.1f} kN")
    print(f"  Thrust (sea level):  {res_design.thrust_sea/1000:.1f} kN")
    print(f"  Isp (vac):           {res_design.isp_vac:.1f} s")
    print(f"  Isp (sea level):     {res_design.isp_sea:.1f} s")
    print(f"  Chamber Pressure:    {config.pc_bar:.1f} bar")
    print(f"  Mixture Ratio:       {config.mr:.2f}")
    print(f"  Expansion Ratio:     {res_design.nozzle.expansion_ratio:.2f}")
    print(f"  Throat Diameter:     {res_design.nozzle.throat_diameter*1000:.2f} mm")
    print(f"  Exit Diameter:       {res_design.nozzle.exit_diameter*1000:.2f} mm")

    # Generate HTML report
    print("\nGenerating HTML report...")
    reporter = HTMLReportGenerator()
    report_path = f"{engine._output_dir}/demo_45kn_design_report.html"
    reporter.generate(res_design, output_path=report_path)
    print(f"Report saved to: {report_path}")

    # 3. Off-Design Runs
    print("\n" + "="*60)
    print("OFF-DESIGN ANALYSIS")
    print("="*60)

    # Case A: Throttled (Low Pc, Low MR)
    print("\n[Case A] Throttled to 67% (40 bar, MR 2.8)")
    res_throttle = engine.analyze(pc_bar=40.0, mr=2.8)
    print(f"  Thrust (vac):        {res_throttle.thrust_vac/1000:.1f} kN")
    print(f"  Isp (vac):           {res_throttle.isp_vac:.1f} s")

    # Case B: Hot Run (High Pc, Design MR)
    print("\n[Case B] Hot Run (80 bar, MR 3.0)")
    res_hot = engine.analyze(pc_bar=80.0, mr=3.0)
    print(f"  Thrust (vac):        {res_hot.thrust_vac/1000:.1f} kN")
    print(f"  Isp (vac):           {res_hot.isp_vac:.1f} s")

    # 4. Throttle Sweep Analysis
    print("\n" + "="*60)
    print("THROTTLE SWEEP ANALYSIS")
    print("="*60)

    throttle_curve = engine.throttle_sweep(
        pc_range=(40.0, 80.0),
        mr_range=(2.5, 3.2),
        num_points=6
    )

    print(f"\nThrottle Curve Generated:")
    print(f"  Number of points: {len(throttle_curve.points)}")
    print(f"  Thrust range: {throttle_curve.points[0].thrust/1000:.1f} - {throttle_curve.points[-1].thrust/1000:.1f} kN")
    print(f"  Isp range: {throttle_curve.points[0].isp_vac:.1f} - {throttle_curve.points[-1].isp_vac:.1f} s")

    # Print throttle table
    print("\nThrottle Points:")
    print(f"{'Pc (bar)':>10} {'MR':>8} {'Thrust (kN)':>14} {'Isp (s)':>10}")
    print("-" * 50)
    for point in throttle_curve.points:
        print(f"{point.pc_bar:>10.1f} {point.mr:>8.2f} {point.thrust/1000:>14.2f} {point.isp_vac:>10.1f}")

    # 5. Visualization
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    try:
        plotter = EngineDashboardPlotter()

        # Create visualization for design point
        fig_design = plotter.create_figure(res_design)
        html_path_design = f"{engine._output_dir}/demo_45kn_design_visualization.html"
        fig_design.write_html(html_path_design)
        print(f"Design visualization saved to: {html_path_design}")

        # Create visualization for throttled case
        fig_throttle = plotter.create_figure(res_throttle)
        html_path_throttle = f"{engine._output_dir}/demo_45kn_throttle_visualization.html"
        fig_throttle.write_html(html_path_throttle)
        print(f"Throttle visualization saved to: {html_path_throttle}")

        # Optionally display
        # fig_design.show()

    except Exception as e:
        print(f"Visualization error: {e}")

    # 6. Export Results Summary
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)

    print(f"\n{'Condition':<20} {'Thrust (kN)':<15} {'Isp (s)':<10} {'Pc (bar)':<12}")
    print("-" * 60)
    print(f"{'Design Point':<20} {res_design.thrust_vac/1000:<15.2f} {res_design.isp_vac:<10.1f} {config.pc_bar:<12.1f}")
    print(f"{'Throttled (67%)':<20} {res_throttle.thrust_vac/1000:<15.2f} {res_throttle.isp_vac:<10.1f} {40.0:<12.1f}")
    print(f"{'Hot Run (133%)':<20} {res_hot.thrust_vac/1000:<15.2f} {res_hot.isp_vac:<10.1f} {80.0:<12.1f}")

    print("\n✅ Demo_45KN Simulation completed successfully.")
