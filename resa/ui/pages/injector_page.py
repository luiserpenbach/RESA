"""Injector Design Page for RESA UI."""
import streamlit as st


def render_injector_page():
    """Render the injector design page."""
    st.title("💧 Swirl Injector Design")

    tab1, tab2, tab3 = st.tabs(["LCSC Design", "GCSC Design", "Cold Flow"])

    with tab1:
        render_lcsc_tab()

    with tab2:
        render_gcsc_tab()

    with tab3:
        render_cold_flow_tab()


def render_lcsc_tab():
    """Render LCSC injector design."""
    st.subheader("Liquid-Centered Swirl Coaxial (LCSC)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Operating Conditions")
        pc = st.number_input("Chamber Pressure [bar]", 5.0, 50.0, 25.0, key="lcsc_pc")
        dp = st.number_input("Pressure Drop [bar]", 2.0, 20.0, 5.0, key="lcsc_dp")
        mdot_ox = st.number_input("Oxidizer Flow [kg/s]", 0.01, 10.0, 0.5, key="lcsc_ox")
        mdot_fuel = st.number_input("Fuel Flow [kg/s]", 0.01, 5.0, 0.125, key="lcsc_fuel")

        st.markdown("### Propellants")
        fuel = st.selectbox("Fuel", ["Ethanol", "RP1", "MMH"], key="lcsc_fuel_sel")
        oxidizer = st.selectbox("Oxidizer", ["N2O", "LOX", "NTO"], key="lcsc_ox_sel")

    with col2:
        st.markdown("### Geometry")
        n_elements = st.number_input("Number of Elements", 1, 50, 12, key="lcsc_n")
        n_ports = st.number_input("Tangential Ports per Element", 2, 8, 4, key="lcsc_ports")
        spray_angle = st.slider("Design Spray Half-Angle [°]", 30, 80, 50, key="lcsc_angle")

        st.markdown("### Constraints")
        min_clearance = st.number_input("Min Clearance [mm]", 0.5, 5.0, 1.0, key="lcsc_clear")

    if st.button("Calculate LCSC", type="primary", key="lcsc_calc"):
        with st.spinner("Sizing injector..."):
            try:
                from resa.addons.injector import LCSCCalculator, InjectorConfig

                config = InjectorConfig(
                    chamber_pressure_pa=pc * 1e5,
                    pressure_drop_pa=dp * 1e5,
                    mass_flow_oxidizer=mdot_ox,
                    mass_flow_fuel=mdot_fuel,
                    num_elements=n_elements,
                    num_fuel_ports=n_ports,
                    spray_half_angle_deg=spray_angle,
                )

                calc = LCSCCalculator()
                result = calc.calculate(config)

                st.success("LCSC sizing complete!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fuel Orifice Dia", f"{result.fuel_orifice_diameter*1000:.2f} mm")
                with col2:
                    st.metric("Swirl Chamber Dia", f"{result.swirl_chamber_diameter*1000:.2f} mm")
                with col3:
                    st.metric("Discharge Coeff", f"{result.discharge_coefficient:.3f}")

            except Exception as e:
                st.error(f"Calculation failed: {e}")


def render_gcsc_tab():
    """Render GCSC injector design."""
    st.subheader("Gas-Centered Swirl Coaxial (GCSC)")
    st.info("GCSC design is similar to LCSC but with gas (oxidizer) in the center.")

    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Chamber Pressure [bar]", 5.0, 50.0, 25.0, key="gcsc_pc")
        st.number_input("Pressure Drop [bar]", 2.0, 20.0, 5.0, key="gcsc_dp")
    with col2:
        st.number_input("Number of Elements", 1, 50, 12, key="gcsc_n")
        st.slider("Design Spray Angle [°]", 30, 80, 50, key="gcsc_angle")

    if st.button("Calculate GCSC", key="gcsc_calc"):
        st.info("GCSC calculation would run here.")


def render_cold_flow_tab():
    """Render cold flow testing predictions."""
    st.subheader("Cold Flow Test Predictions")

    st.markdown("Predict performance with water/nitrogen for testing.")

    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Fuel Simulant", ["Water", "Ethanol (diluted)"], key="cf_fuel")
        st.selectbox("Oxidizer Simulant", ["Nitrogen", "Air", "Water"], key="cf_ox")
    with col2:
        st.number_input("Test Pressure [bar]", 2.0, 20.0, 10.0, key="cf_p")
        st.number_input("Test Temperature [°C]", 10.0, 40.0, 20.0, key="cf_t")

    if st.button("Predict Cold Flow", key="cf_calc"):
        st.info("Cold flow predictions would be calculated here.")
