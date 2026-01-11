"""Igniter Design Page for RESA UI."""
import streamlit as st


def render_igniter_page():
    """Render the torch igniter design page."""
    st.title("🔥 Torch Igniter Design")

    tab1, tab2, tab3 = st.tabs(["Design", "Analysis", "Report"])

    with tab1:
        render_design_tab()

    with tab2:
        render_analysis_tab()

    with tab3:
        render_report_tab()


def render_design_tab():
    """Render igniter design inputs."""
    st.subheader("Igniter Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Operating Conditions")
        pc = st.number_input("Chamber Pressure [bar]", 1.0, 30.0, 10.0, key="ign_pc")
        mr = st.number_input("Mixture Ratio", 1.0, 10.0, 5.0, key="ign_mr")
        mdot = st.number_input("Total Mass Flow [g/s]", 1.0, 100.0, 10.0, key="ign_mdot")

        st.markdown("### Propellants")
        fuel = st.selectbox("Fuel", ["Ethanol", "Methane", "Propane"], key="ign_fuel")
        oxidizer = st.selectbox("Oxidizer", ["N2O", "GOX"], key="ign_ox")

    with col2:
        st.markdown("### Feed Conditions")
        st.markdown("**Fuel**")
        p_fuel = st.number_input("Feed Pressure [bar]", 5.0, 50.0, 20.0, key="ign_p_fuel")
        t_fuel = st.number_input("Temperature [K]", 250.0, 350.0, 293.0, key="ign_t_fuel")

        st.markdown("**Oxidizer**")
        p_ox = st.number_input("Feed Pressure [bar]", 5.0, 70.0, 50.0, key="ign_p_ox")
        t_ox = st.number_input("Temperature [K]", 250.0, 320.0, 290.0, key="ign_t_ox")

        st.markdown("### Design Parameters")
        l_star = st.number_input("L* [mm]", 200.0, 1500.0, 800.0, key="ign_lstar")
        eps = st.number_input("Expansion Ratio", 1.5, 10.0, 3.0, key="ign_eps")

    if st.button("🔥 Design Igniter", type="primary"):
        with st.spinner("Running CEA and sizing..."):
            try:
                from resa.addons.igniter import IgniterDesigner, IgniterConfig

                config = IgniterConfig(
                    chamber_pressure_bar=pc,
                    mixture_ratio=mr,
                    total_mass_flow_kg_s=mdot / 1000,
                    fuel=fuel.lower(),
                    oxidizer="NitrousOxide" if oxidizer == "N2O" else "GOX",
                    fuel_feed_pressure_bar=p_fuel,
                    fuel_feed_temperature_k=t_fuel,
                    oxidizer_feed_pressure_bar=p_ox,
                    oxidizer_feed_temperature_k=t_ox,
                    l_star_m=l_star / 1000,
                    expansion_ratio=eps,
                )

                designer = IgniterDesigner()
                result = designer.design(config)

                st.session_state.igniter_result = result
                st.success("Igniter design complete!")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("C*", f"{result.cstar:.0f} m/s")
                with col2:
                    st.metric("Isp", f"{result.isp:.1f} s")
                with col3:
                    st.metric("Throat Dia", f"{result.throat_diameter*1000:.2f} mm")
                with col4:
                    st.metric("Heat Power", f"{result.heat_power/1000:.1f} kW")

            except Exception as e:
                st.error(f"Design failed: {e}")


def render_analysis_tab():
    """Render igniter analysis."""
    st.subheader("Operating Envelope")

    if not st.session_state.get('igniter_result'):
        st.warning("Run the design first.")
        return

    st.markdown("### Mixture Ratio Sweep")
    mr_min = st.slider("MR Min", 2.0, 5.0, 3.0)
    mr_max = st.slider("MR Max", 5.0, 10.0, 7.0)

    if st.button("Generate Envelope"):
        st.info("Would generate MR sweep analysis here.")


def render_report_tab():
    """Render igniter report."""
    st.subheader("Generate Report")

    if not st.session_state.get('igniter_result'):
        st.warning("Run the design first.")
        return

    if st.button("📄 Generate Report"):
        st.success("Report generated! (Demo)")
