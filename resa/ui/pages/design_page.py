"""
Engine Design Page for RESA UI.

Provides the main engine design workflow with:
- Configuration input
- Design execution
- Results visualization
- Report generation
"""
import streamlit as st
import numpy as np
from datetime import datetime


def render_design_page():
    """Render the engine design page."""
    st.title("🔧 Engine Design")

    # Tabs for workflow
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Configuration",
        "▶️ Run Design",
        "📊 Results",
        "📄 Report"
    ])

    with tab1:
        render_configuration_tab()

    with tab2:
        render_run_tab()

    with tab3:
        render_results_tab()

    with tab4:
        render_report_tab()


def render_configuration_tab():
    """Render the configuration input tab."""
    st.subheader("Engine Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### General")
        engine_name = st.text_input("Engine Name", value="Phoenix-1")
        designer = st.text_input("Designer", value="")

        st.markdown("### Propellants")
        fuel = st.selectbox("Fuel", ["Ethanol90", "Ethanol", "RP1", "Methane", "LH2"])
        oxidizer = st.selectbox("Oxidizer", ["N2O", "LOX", "H2O2"])

        st.markdown("### Operating Point")
        thrust = st.number_input("Target Thrust [N]", min_value=10.0, max_value=1e6, value=2200.0)
        pc = st.number_input("Chamber Pressure [bar]", min_value=1.0, max_value=300.0, value=25.0)
        mr = st.number_input("Mixture Ratio (O/F)", min_value=0.5, max_value=15.0, value=4.0)
        eta_c = st.slider("Combustion Efficiency", min_value=0.8, max_value=1.0, value=0.95)

    with col2:
        st.markdown("### Nozzle Geometry")
        expansion_mode = st.radio("Expansion Ratio", ["Optimal (sea level)", "Optimal (altitude)", "Fixed"])
        if expansion_mode == "Fixed":
            eps = st.number_input("Expansion Ratio", min_value=1.5, max_value=100.0, value=10.0)
        elif expansion_mode == "Optimal (altitude)":
            p_exit = st.number_input("Design Altitude Pressure [bar]", min_value=0.001, max_value=1.0, value=0.1)
            eps = 0.0
        else:
            p_exit = 1.013
            eps = 0.0

        L_star = st.number_input("L* (Characteristic Length) [mm]", min_value=300.0, max_value=3000.0, value=1100.0)
        cr = st.number_input("Contraction Ratio", min_value=2.0, max_value=20.0, value=8.0)
        bell_frac = st.slider("Bell Fraction", min_value=0.6, max_value=1.0, value=0.8)

        st.markdown("### Cooling System")
        coolant = st.selectbox("Coolant", ["N2O (self-pressurizing)", "RP1", "Ethanol", "Water"])
        coolant_map = {
            "N2O (self-pressurizing)": "REFPROP::NitrousOxide",
            "RP1": "INCOMP::DowQ",
            "Ethanol": "Ethanol",
            "Water": "Water"
        }
        mode = st.radio("Flow Mode", ["Counter-flow", "Co-flow"])

        st.markdown("### Channel Geometry (at Throat)")
        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            w_ch = st.number_input("Channel Width [mm]", min_value=0.3, max_value=10.0, value=1.0)
            h_ch = st.number_input("Channel Height [mm]", min_value=0.3, max_value=10.0, value=0.75)
        with col_ch2:
            w_rib = st.number_input("Rib Width [mm]", min_value=0.3, max_value=5.0, value=0.6)
            t_wall = st.number_input("Wall Thickness [mm]", min_value=0.3, max_value=5.0, value=0.5)

        st.markdown("### Coolant Inlet")
        p_in = st.number_input("Inlet Pressure [bar]", min_value=10.0, max_value=200.0, value=60.0)
        t_in = st.number_input("Inlet Temperature [K]", min_value=200.0, max_value=400.0, value=290.0)

    # Save configuration
    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        try:
            from resa.core.config import EngineConfig

            config = EngineConfig(
                engine_name=engine_name,
                version="1.0",
                designer=designer,
                fuel=fuel,
                oxidizer=oxidizer,
                thrust_n=thrust,
                pc_bar=pc,
                mr=mr,
                eff_combustion=eta_c,
                expansion_ratio=eps if expansion_mode == "Fixed" else 0.0,
                p_exit_bar=p_exit if expansion_mode != "Fixed" else 1.013,
                L_star=L_star,
                contraction_ratio=cr,
                bell_fraction=bell_frac,
                coolant_name=coolant_map.get(coolant, "REFPROP::NitrousOxide"),
                cooling_mode=mode.lower().replace("-", "_"),
                channel_width_throat=w_ch / 1000.0,
                channel_height=h_ch / 1000.0,
                rib_width_throat=w_rib / 1000.0,
                wall_thickness=t_wall / 1000.0,
                coolant_p_in_bar=p_in,
                coolant_t_in_k=t_in,
            )

            st.session_state.engine_config = config
            st.success(f"Configuration saved: {engine_name}")

            # Add to history
            if 'analysis_history' not in st.session_state:
                st.session_state.analysis_history = []
            st.session_state.analysis_history.append(
                f"{datetime.now().strftime('%H:%M')} - Config saved: {engine_name}"
            )

        except Exception as e:
            st.error(f"Error creating config: {e}")


def render_run_tab():
    """Render the design execution tab."""
    st.subheader("Run Design Analysis")

    if not st.session_state.get('engine_config'):
        st.warning("Please configure the engine first in the Configuration tab.")
        return

    config = st.session_state.engine_config
    st.info(f"Ready to run design for **{config.engine_name}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Thrust Target", f"{config.thrust_n:.0f} N")
    with col2:
        st.metric("Chamber Pressure", f"{config.pc_bar:.1f} bar")
    with col3:
        st.metric("Mixture Ratio", f"{config.mr:.2f}")

    st.divider()

    if st.button("🚀 Run Full Design", type="primary", use_container_width=True):
        with st.spinner("Running combustion analysis..."):
            progress = st.progress(0)

            try:
                # Simulate progress steps
                progress.progress(10, "Initializing solvers...")

                # Import and run
                from resa.core.engine import Engine

                progress.progress(20, "Running CEA combustion...")
                engine = Engine(config)

                progress.progress(40, "Generating nozzle geometry...")
                progress.progress(60, "Analyzing gas dynamics...")
                progress.progress(80, "Running cooling analysis...")

                result = engine.design()

                progress.progress(100, "Complete!")

                st.session_state.design_result = result

                st.success("Design analysis complete!")

                # Show quick results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Isp (vac)", f"{result.isp_vac:.1f} s")
                with col2:
                    st.metric("Thrust (vac)", f"{result.thrust_vac:.0f} N")
                with col3:
                    st.metric("Throat Dia", f"{result.dt_mm:.2f} mm")
                with col4:
                    st.metric("Exit Dia", f"{result.de_mm:.2f} mm")

                # Add to history
                st.session_state.analysis_history.append(
                    f"{datetime.now().strftime('%H:%M')} - Design run: {config.engine_name}, Isp={result.isp_vac:.1f}s"
                )

            except Exception as e:
                st.error(f"Design failed: {e}")
                import traceback
                st.code(traceback.format_exc())


def render_results_tab():
    """Render the results visualization tab."""
    st.subheader("Design Results")

    if not st.session_state.get('design_result'):
        st.warning("Run the design analysis first.")
        return

    result = st.session_state.design_result

    # Performance summary
    st.markdown("### Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Isp (vacuum)", f"{result.isp_vac:.1f} s")
        st.metric("Isp (sea level)", f"{result.isp_sea:.1f} s")
    with col2:
        st.metric("Thrust (vacuum)", f"{result.thrust_vac:.0f} N")
        st.metric("Thrust (sea level)", f"{result.thrust_sea:.0f} N")
    with col3:
        st.metric("Mass Flow", f"{result.massflow_total:.4f} kg/s")
        st.metric("Expansion Ratio", f"{result.expansion_ratio:.2f}")
    with col4:
        st.metric("Throat Diameter", f"{result.dt_mm:.2f} mm")
        st.metric("Exit Diameter", f"{result.de_mm:.2f} mm")

    st.divider()

    # Visualizations
    st.markdown("### Visualizations")

    viz_tabs = st.tabs(["Dashboard", "Cross-Section", "3D View"])

    with viz_tabs[0]:
        try:
            from resa.visualization.engine_plots import EngineDashboardPlotter

            plotter = EngineDashboardPlotter()
            fig = plotter.create_figure(result)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not render dashboard: {e}")

    with viz_tabs[1]:
        try:
            from resa.visualization.engine_plots import CrossSectionPlotter

            if result.channel_geometry:
                plotter = CrossSectionPlotter()
                throat_idx = np.argmin(result.nozzle_geometry.y_full)
                fig = plotter.create_figure(result.channel_geometry, station_idx=throat_idx)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Channel geometry not available")
        except Exception as e:
            st.error(f"Could not render cross-section: {e}")

    with viz_tabs[2]:
        try:
            from resa.visualization.engine_3d import Engine3DViewer

            viewer = Engine3DViewer()
            fig = viewer.create_figure(result.nozzle_geometry)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info("3D visualization requires complete geometry data")


def render_report_tab():
    """Render the report generation tab."""
    st.subheader("Generate Report")

    if not st.session_state.get('design_result'):
        st.warning("Run the design analysis first.")
        return

    result = st.session_state.design_result
    config = st.session_state.engine_config

    st.markdown("### Report Options")

    col1, col2 = st.columns(2)
    with col1:
        include_plots = st.checkbox("Include performance plots", value=True)
        include_geometry = st.checkbox("Include geometry details", value=True)
        include_cooling = st.checkbox("Include cooling analysis", value=True)
    with col2:
        include_3d = st.checkbox("Include 3D visualization", value=False)
        include_raw_data = st.checkbox("Include raw data tables", value=False)

    report_format = st.selectbox("Output Format", ["HTML", "PDF (requires weasyprint)"])

    if st.button("📄 Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            try:
                from resa.reporting.html_report import HTMLReportGenerator

                generator = HTMLReportGenerator()
                html = generator.generate(result)

                st.success("Report generated!")

                # Download button
                st.download_button(
                    "⬇️ Download HTML Report",
                    html,
                    file_name=f"{config.engine_name}_report.html",
                    mime="text/html"
                )

                # Preview
                with st.expander("Preview Report"):
                    st.components.v1.html(html, height=800, scrolling=True)

            except Exception as e:
                st.error(f"Report generation failed: {e}")
