"""
Injector Design Page - Swirl injector sizing and analysis
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go


def render_injector_page():
    """Render the injector design page."""
    st.title("💉 Injector Design")
    
    st.markdown("""
    Design and analyze swirl injectors for liquid rocket engines.
    Supports LCSC (Liquid-Centered Swirl Coaxial) configuration.
    """)
    
    tab_size, tab_perf, tab_cold = st.tabs([
        "📐 Sizing",
        "📈 Performance",
        "🧪 Cold Flow"
    ])
    
    with tab_size:
        render_sizing_tab()
    
    with tab_perf:
        st.info("Performance analysis connects to existing swirl injector code.")
    
    with tab_cold:
        st.info("Cold flow test correlation with water/nitrogen.")


def render_sizing_tab():
    """Render injector sizing inputs."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Operating Conditions")
        
        pc_bar = st.number_input("Chamber Pressure [bar]", value=25.0, min_value=5.0)
        dp_bar = st.number_input("Injector ΔP [bar]", value=5.0, min_value=1.0)
        
        st.markdown("### Mass Flows")
        
        mdot_fuel = st.number_input("Fuel Mass Flow [kg/s]", value=0.2, min_value=0.01)
        mdot_ox = st.number_input("Oxidizer Mass Flow [kg/s]", value=0.8, min_value=0.01)
        n_elements = st.number_input("Number of Elements", value=3, min_value=1, max_value=20)
        
    with col2:
        st.markdown("### Injector Parameters")
        
        alpha_deg = st.slider("Target Spray Half-Angle [°]", 40, 70, 60)
        n_ports = st.slider("Tangential Ports per Element", 2, 6, 3)
        
        st.markdown("### Fluid Properties")
        
        rho_fuel = st.number_input("Fuel Density [kg/m³]", value=800.0)
        rho_ox = st.number_input("Oxidizer Density [kg/m³]", value=750.0)
        
        if st.button("🔧 Size Injector", type="primary"):
            results = size_swirl_injector(
                pc_bar, dp_bar, mdot_fuel, mdot_ox, n_elements,
                alpha_deg, n_ports, rho_fuel, rho_ox
            )
            
            display_injector_results(results)


def size_swirl_injector(pc, dp, mdot_f, mdot_ox, n_el, alpha, n_ports, rho_f, rho_ox):
    """Simple swirl injector sizing."""
    
    # Per-element flows
    mdot_f_el = mdot_f / n_el
    mdot_ox_el = mdot_ox / n_el
    
    # X factor from spray angle
    X = 0.0042 * alpha ** 1.2714
    
    # Discharge coefficient
    Cd = np.sqrt(((1 - X) ** 3) / (1 + X))
    
    # Orifice area
    dp_pa = dp * 1e5
    A_orifice = mdot_f_el / (Cd * np.sqrt(2 * rho_f * dp_pa))
    d_orifice = np.sqrt(4 * A_orifice / np.pi) * 1000  # mm
    
    # Swirl chamber
    d_chamber = 3.3 * d_orifice
    
    # Inlet ports
    Cd_port = np.sqrt(X ** 3 / (2 - X))
    A_port_total = mdot_f_el / (Cd_port * np.sqrt(2 * rho_f * dp_pa))
    A_port = A_port_total / n_ports
    d_port = np.sqrt(4 * A_port / np.pi) * 1000  # mm
    
    # Film thickness
    t_film = d_orifice / 2 * (1 - np.sqrt(X))
    
    return {
        'd_orifice_mm': d_orifice,
        'd_chamber_mm': d_chamber,
        'd_port_mm': d_port,
        'Cd': Cd,
        'X': X,
        't_film_mm': t_film,
        'n_elements': n_el,
        'n_ports': n_ports,
        'mdot_f_element': mdot_f_el,
        'mdot_ox_element': mdot_ox_el
    }


def display_injector_results(res: dict):
    """Display injector sizing results."""
    st.markdown("---")
    st.markdown("### Sizing Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Orifice Diameter", f"{res['d_orifice_mm']:.2f} mm")
        st.metric("Chamber Diameter", f"{res['d_chamber_mm']:.2f} mm")
    
    with col2:
        st.metric("Inlet Port Diameter", f"{res['d_port_mm']:.2f} mm")
        st.metric("Film Thickness", f"{res['t_film_mm']:.3f} mm")
    
    with col3:
        st.metric("Discharge Coefficient", f"{res['Cd']:.3f}")
        st.metric("X Factor", f"{res['X']:.4f}")
    
    # Simple visualization
    fig = go.Figure()
    
    # Draw simplified injector cross-section
    theta = np.linspace(0, 2*np.pi, 100)
    r_orifice = res['d_orifice_mm'] / 2
    r_chamber = res['d_chamber_mm'] / 2
    
    # Chamber circle
    fig.add_trace(go.Scatter(
        x=r_chamber * np.cos(theta),
        y=r_chamber * np.sin(theta),
        mode='lines',
        name='Swirl Chamber',
        line=dict(color='blue', width=2)
    ))
    
    # Orifice circle
    fig.add_trace(go.Scatter(
        x=r_orifice * np.cos(theta),
        y=r_orifice * np.sin(theta),
        mode='lines',
        name='Orifice',
        line=dict(color='red', width=3)
    ))
    
    # Air core
    r_aircore = r_orifice * np.sqrt(res['X'])
    fig.add_trace(go.Scatter(
        x=r_aircore * np.cos(theta),
        y=r_aircore * np.sin(theta),
        mode='lines',
        name='Air Core',
        line=dict(color='cyan', width=2, dash='dash'),
        fill='toself',
        fillcolor='rgba(0, 255, 255, 0.2)'
    ))
    
    fig.update_layout(
        title="Injector Cross-Section (Schematic)",
        xaxis_title="mm",
        yaxis_title="mm",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(f"Configuration: {res['n_elements']} elements × {res['n_ports']} ports each")
