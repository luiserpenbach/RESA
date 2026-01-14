"""
Demo Simulation: 50N Engine (C2H6/N2O)
Updated to use the new resa package (v2.0)
"""
from resa import Engine, EngineConfig
from resa.visualization import EngineDashboardPlotter
from resa.reporting import HTMLReportGenerator

if __name__ == "__main__":
    # 1. Define Design Point Configuration
    config = EngineConfig(
        engine_name="Demo-50N",
        fuel="C2H6",
        oxidizer="N2O",
        thrust_n=50,
        pc_bar=8,
        mr=7.0,
        p_exit_bar=1e-3,
        contraction_ratio=15,
        expansion_ratio=0,  # Will be calculated for optimal expansion
        L_star=900,

        # Cooling Geometry
        channel_width_throat=0.5e-3,
        channel_height=0.75e-3,
        rib_width_throat=0.5e-3,
        wall_thickness=0.4e-3,

        # Coolant
        coolant_name="REFPROP::NitrousOxide",
        coolant_p_in_bar=55.0,
        coolant_t_in_k=293.15
    )

    engine = Engine(config, output_dir="output/output-demo50N")

    # 2. Design Point Run
    print("\n" + "="*60)
    print("DESIGN POINT ANALYSIS")
    print("="*60)
    res_design = engine.design()

    print(f"\nDesign Results:")
    print(f"  Thrust (vac):        {res_design.thrust_vac:.1f} N")
    print(f"  Isp (vac):           {res_design.isp_vac:.1f} s")
    print(f"  Chamber Pressure:    {config.pc_bar:.1f} bar")
    print(f"  Mixture Ratio:       {config.mr:.2f}")
    print(f"  Expansion Ratio:     {res_design.nozzle.expansion_ratio:.2f}")
    print(f"  Throat Diameter:     {res_design.nozzle.throat_diameter*1000:.2f} mm")

    # Generate HTML report
    print("\nGenerating HTML report...")
    reporter = HTMLReportGenerator()
    report_path = f"{engine._output_dir}/demo_50n_design_report.html"
    reporter.generate(res_design, output_path=report_path)
    print(f"Report saved to: {report_path}")

    # 3. Off-Design Runs
    print("\n" + "="*60)
    print("OFF-DESIGN ANALYSIS: Throttled Condition")
    print("="*60)
    res_throttle = engine.analyze(pc_bar=4.0, mr=4.0)

    print(f"\nThrottled Results:")
    print(f"  Thrust (vac):        {res_throttle.thrust_vac:.1f} N")
    print(f"  Isp (vac):           {res_throttle.isp_vac:.1f} s")
    print(f"  Chamber Pressure:    {res_throttle.combustion.chamber_pressure/1e5:.1f} bar")
    print(f"  Throttle Level:      ~50%")

    # 4. Visualization
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    try:
        plotter = EngineDashboardPlotter()
        fig = plotter.create_figure(res_design)

        # Save as HTML
        html_path = f"{engine._output_dir}/demo_50n_visualization.html"
        fig.write_html(html_path)
        print(f"Interactive visualization saved to: {html_path}")

        # Optionally display
        # fig.show()

    except Exception as e:
        print(f"Visualization error: {e}")

    # 5. Throttle Sweep
    print("\n" + "="*60)
    print("THROTTLE SWEEP ANALYSIS")
    print("="*60)

    throttle_curve = engine.throttle_sweep(
        pc_range=(4.0, 8.0),
        mr_range=(4.0, 7.0),
        num_points=5
    )

    print(f"\nThrottle Curve Generated:")
    print(f"  Number of points: {len(throttle_curve.points)}")
    print(f"  Thrust range: {throttle_curve.points[0].thrust:.0f} - {throttle_curve.points[-1].thrust:.0f} N")
    print(f"  Isp range: {throttle_curve.points[0].isp_vac:.1f} - {throttle_curve.points[-1].isp_vac:.1f} s")

    print("\n✅ Demo_50N Simulation completed successfully.")
