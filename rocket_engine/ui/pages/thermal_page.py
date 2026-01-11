"""
Thermal Analysis Page - Detailed cooling analysis
"""
import streamlit as st


def render_thermal_page():
    """Render the thermal analysis page."""
    st.title("🌡️ Thermal Analysis")
    
    st.markdown("""
    Detailed regenerative cooling analysis including:
    - Wall temperature profiles
    - Heat flux distributions
    - Coolant state evolution
    - Phase diagram tracking
    """)
    
    if not st.session_state.get('design_result'):
        st.warning("⚠️ Please complete an engine design first to view thermal analysis.")
        return
    
    result = st.session_state.design_result
    
    st.markdown("---")
    st.markdown("### Cooling Summary")
    
    if result.cooling:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Max Wall Temp", f"{result.cooling.max_wall_temp:.0f} K")
        with col2:
            st.metric("Max Heat Flux", f"{result.cooling.max_heat_flux/1e6:.1f} MW/m²")
        with col3:
            st.metric("Pressure Drop", f"{result.cooling.pressure_drop/1e5:.1f} bar")
        with col4:
            outlet_temp = result.cooling.T_coolant[-1] if len(result.cooling.T_coolant) > 0 else 0
            st.metric("Coolant Outlet T", f"{outlet_temp:.0f} K")
    
    st.info("Full thermal analysis visualization connects to `rocket_engine.analysis.fluid_state`.")
