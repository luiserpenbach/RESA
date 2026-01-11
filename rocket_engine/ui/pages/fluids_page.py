"""
Fluid Properties Page - Interactive thermodynamic property explorer
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Try to import CoolProp
try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False


# Fluid definitions
FLUIDS = {
    "Nitrous Oxide (N2O)": "NitrousOxide",
    "Oxygen (LOX)": "Oxygen", 
    "Ethanol": "Ethanol",
    "Water": "Water",
    "Methane": "Methane",
    "Propane": "Propane",
    "Carbon Dioxide": "CO2",
    "Nitrogen": "Nitrogen",
}


def render_fluids_page():
    """Render the fluid properties explorer page."""
    st.title("🔬 Fluid Properties Explorer")
    
    if not COOLPROP_AVAILABLE:
        st.error("CoolProp is not installed. Please run: `pip install CoolProp`")
        return
    
    st.markdown("""
    Explore thermodynamic properties of common rocket propellants and coolants.
    This tool uses CoolProp for high-accuracy real fluid properties.
    """)
    
    # Fluid selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fluid_display = st.selectbox(
            "Select Fluid",
            list(FLUIDS.keys()),
            index=0
        )
        fluid_name = FLUIDS[fluid_display]
    
    with col2:
        # Get critical properties
        try:
            T_crit = CP.PropsSI('Tcrit', fluid_name)
            P_crit = CP.PropsSI('pcrit', fluid_name)
            rho_crit = CP.PropsSI('rhocrit', fluid_name)
            
            st.markdown("**Critical Point**")
            st.caption(f"T_c = {T_crit:.1f} K ({T_crit-273.15:.1f} °C)")
            st.caption(f"P_c = {P_crit/1e5:.2f} bar")
            st.caption(f"ρ_c = {rho_crit:.1f} kg/m³")
        except:
            st.warning("Could not get critical properties")
            T_crit = 400
            P_crit = 100e5
            rho_crit = 400
    
    st.markdown("---")
    
    # Create tabs for different views
    tab_pt, tab_sat, tab_calc, tab_orifice = st.tabs([
        "📊 P-T Diagram",
        "🌡️ Saturation Properties",
        "🔢 Point Calculator",
        "💧 Orifice Flow Models"
    ])
    
    with tab_pt:
        render_pt_diagram(fluid_name, T_crit, P_crit, rho_crit)
    
    with tab_sat:
        render_saturation_table(fluid_name, T_crit)
    
    with tab_calc:
        render_point_calculator(fluid_name)
    
    with tab_orifice:
        render_orifice_models(fluid_name, T_crit, P_crit)


def render_pt_diagram(fluid_name: str, T_crit: float, P_crit: float, rho_crit: float):
    """Render pressure-temperature phase diagram with density contours."""
    st.markdown("### Pressure-Temperature Phase Diagram")
    
    col_plot, col_ctrl = st.columns([3, 1])
    
    with col_ctrl:
        st.markdown("**Plot Controls**")
        
        try:
            T_triple = CP.PropsSI('T_triple', fluid_name)
        except:
            T_triple = 180.0
        
        T_min = st.number_input("T min [K]", value=float(T_triple + 5), min_value=100.0)
        T_max = st.number_input("T max [K]", value=float(min(T_crit * 1.15, 500.0)))
        P_max_bar = st.number_input("P max [bar]", value=float(min(P_crit/1e5 * 1.3, 150.0)))
        
        show_density = st.checkbox("Show Density Contours", value=True)
        
        st.markdown("**Mark Operating Point**")
        mark_point = st.checkbox("Show Point", value=False)
        if mark_point:
            op_T = st.number_input("Temperature [K]", value=290.0, key="pt_op_T")
            op_P = st.number_input("Pressure [bar]", value=50.0, key="pt_op_P")
    
    with col_plot:
        # Generate saturation curve
        T_sat = np.linspace(T_triple + 1, T_crit - 0.5, 100)
        P_sat = []
        
        for T in T_sat:
            try:
                P_sat.append(CP.PropsSI('P', 'T', T, 'Q', 0, fluid_name))
            except:
                P_sat.append(np.nan)
        
        P_sat = np.array(P_sat)
        
        fig = go.Figure()
        
        # Add density contours as background
        if show_density:
            resolution = 50
            T_grid = np.linspace(T_min, T_max, resolution)
            P_grid = np.linspace(1e5, P_max_bar * 1e5, resolution)
            T_mesh, P_mesh = np.meshgrid(T_grid, P_grid)
            
            rho_grid = np.zeros_like(T_mesh)
            for i in range(resolution):
                for j in range(resolution):
                    try:
                        rho_grid[i, j] = CP.PropsSI('D', 'T', T_grid[j], 'P', P_grid[i], fluid_name)
                    except:
                        rho_grid[i, j] = np.nan
            
            fig.add_trace(go.Contour(
                x=T_grid,
                y=P_grid / 1e5,
                z=rho_grid,
                colorscale='Viridis',
                opacity=0.6,
                name='Density',
                colorbar=dict(title='ρ [kg/m³]', x=1.02),
                contours=dict(showlabels=True, labelfont=dict(size=10, color='white'))
            ))
        
        # Saturation line
        fig.add_trace(go.Scatter(
            x=T_sat,
            y=P_sat / 1e5,
            mode='lines',
            name='Saturation Line',
            line=dict(color='white', width=3)
        ))
        
        # Critical point
        fig.add_trace(go.Scatter(
            x=[T_crit],
            y=[P_crit / 1e5],
            mode='markers',
            name='Critical Point',
            marker=dict(size=12, color='red', symbol='x')
        ))
        
        # Operating point
        if mark_point:
            fig.add_trace(go.Scatter(
                x=[op_T],
                y=[op_P],
                mode='markers',
                name='Operating Point',
                marker=dict(size=15, color='yellow', symbol='star')
            ))
        
        # Phase region labels
        fig.add_annotation(x=T_min + 20, y=P_max_bar * 0.9, text="Liquid",
                          showarrow=False, font=dict(size=14, color='white'))
        fig.add_annotation(x=T_max - 20, y=5, text="Vapor",
                          showarrow=False, font=dict(size=14, color='white'))
        fig.add_annotation(x=T_crit + 10, y=P_crit/1e5 + 10, text="Supercritical",
                          showarrow=False, font=dict(size=12, color='white'))
        
        fig.update_layout(
            title=f"{fluid_name} Phase Diagram",
            xaxis_title="Temperature [K]",
            yaxis_title="Pressure [bar]",
            height=500,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        fig.update_xaxes(range=[T_min, T_max])
        fig.update_yaxes(range=[0, P_max_bar])
        
        st.plotly_chart(fig, use_container_width=True)


def render_saturation_table(fluid_name: str, T_crit: float):
    """Render saturation properties table."""
    st.markdown("### Saturation Properties")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        try:
            T_triple = CP.PropsSI('T_triple', fluid_name)
        except:
            T_triple = 180.0
        
        T_start = st.number_input("Start Temp [K]", value=float(T_triple + 10))
        T_end = st.number_input("End Temp [K]", value=float(T_crit - 5))
        n_points = st.slider("Number of Points", 5, 50, 20)
    
    with col2:
        temperatures = np.linspace(T_start, T_end, n_points)
        
        data = []
        for T in temperatures:
            try:
                P_sat = CP.PropsSI('P', 'T', T, 'Q', 0, fluid_name)
                rho_l = CP.PropsSI('D', 'T', T, 'Q', 0, fluid_name)
                rho_v = CP.PropsSI('D', 'T', T, 'Q', 1, fluid_name)
                h_l = CP.PropsSI('H', 'T', T, 'Q', 0, fluid_name)
                h_v = CP.PropsSI('H', 'T', T, 'Q', 1, fluid_name)
                h_fg = h_v - h_l
                
                data.append({
                    'T [K]': f"{T:.1f}",
                    'T [°C]': f"{T-273.15:.1f}",
                    'P [bar]': f"{P_sat/1e5:.2f}",
                    'ρ_l [kg/m³]': f"{rho_l:.1f}",
                    'ρ_v [kg/m³]': f"{rho_v:.2f}",
                    'h_fg [kJ/kg]': f"{h_fg/1000:.1f}",
                })
            except:
                pass
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"{fluid_name}_saturation.csv",
            "text/csv"
        )


def render_point_calculator(fluid_name: str):
    """Interactive single-point property calculator."""
    st.markdown("### Point Property Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Input State")
        
        input_type = st.radio(
            "Input Variables",
            ["P-T (Pressure, Temperature)", "P-H (Pressure, Enthalpy)", "P-Q (Pressure, Quality)"],
            horizontal=True
        )
        
        if input_type == "P-T (Pressure, Temperature)":
            P_input = st.number_input("Pressure [bar]", value=50.0, min_value=0.1)
            T_input = st.number_input("Temperature [K]", value=290.0, min_value=100.0)
            
            try:
                state = get_state_PT(fluid_name, P_input * 1e5, T_input)
            except Exception as e:
                st.error(f"Error: {e}")
                state = None
                
        elif input_type == "P-H (Pressure, Enthalpy)":
            P_input = st.number_input("Pressure [bar]", value=50.0, min_value=0.1)
            H_input = st.number_input("Enthalpy [kJ/kg]", value=200.0)
            
            try:
                state = get_state_PH(fluid_name, P_input * 1e5, H_input * 1000)
            except Exception as e:
                st.error(f"Error: {e}")
                state = None
                
        else:  # P-Q
            P_input = st.number_input("Pressure [bar]", value=50.0, min_value=0.1)
            Q_input = st.slider("Quality (x)", 0.0, 1.0, 0.5, 0.01)
            
            try:
                state = get_state_PQ(fluid_name, P_input * 1e5, Q_input)
            except Exception as e:
                st.error(f"Error: {e}")
                state = None
    
    with col2:
        st.markdown("#### Calculated Properties")
        
        if state:
            # Display in organized groups
            st.markdown("**Thermodynamic**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Temperature", f"{state['T']:.2f} K")
                st.metric("Pressure", f"{state['P']/1e5:.2f} bar")
                st.metric("Density", f"{state['rho']:.2f} kg/m³")
            with col_b:
                st.metric("Enthalpy", f"{state['h']/1000:.2f} kJ/kg")
                st.metric("Entropy", f"{state['s']:.1f} J/(kg·K)")
                st.metric("Quality", f"{state['Q']:.3f}" if state['Q'] >= 0 else "Single Phase")
            
            st.markdown("**Transport**")
            col_c, col_d = st.columns(2)
            with col_c:
                st.metric("Viscosity", f"{state['mu']*1e6:.2f} µPa·s")
                st.metric("Conductivity", f"{state['k']*1000:.2f} mW/(m·K)")
            with col_d:
                st.metric("Cp", f"{state['cp']:.1f} J/(kg·K)")
                st.metric("Speed of Sound", f"{state['a']:.1f} m/s")
        else:
            st.info("Enter valid inputs to calculate properties")


def get_state_PT(fluid: str, P: float, T: float) -> dict:
    """Get state from P-T inputs."""
    return {
        'T': T,
        'P': P,
        'rho': CP.PropsSI('D', 'P', P, 'T', T, fluid),
        'h': CP.PropsSI('H', 'P', P, 'T', T, fluid),
        's': CP.PropsSI('S', 'P', P, 'T', T, fluid),
        'Q': CP.PropsSI('Q', 'P', P, 'T', T, fluid),
        'mu': CP.PropsSI('V', 'P', P, 'T', T, fluid),
        'k': CP.PropsSI('L', 'P', P, 'T', T, fluid),
        'cp': CP.PropsSI('Cpmass', 'P', P, 'T', T, fluid),
        'a': CP.PropsSI('A', 'P', P, 'T', T, fluid),
    }


def get_state_PH(fluid: str, P: float, H: float) -> dict:
    """Get state from P-H inputs."""
    return {
        'T': CP.PropsSI('T', 'P', P, 'H', H, fluid),
        'P': P,
        'rho': CP.PropsSI('D', 'P', P, 'H', H, fluid),
        'h': H,
        's': CP.PropsSI('S', 'P', P, 'H', H, fluid),
        'Q': CP.PropsSI('Q', 'P', P, 'H', H, fluid),
        'mu': CP.PropsSI('V', 'P', P, 'H', H, fluid),
        'k': CP.PropsSI('L', 'P', P, 'H', H, fluid),
        'cp': CP.PropsSI('Cpmass', 'P', P, 'H', H, fluid),
        'a': CP.PropsSI('A', 'P', P, 'H', H, fluid),
    }


def get_state_PQ(fluid: str, P: float, Q: float) -> dict:
    """Get state from P-Q inputs."""
    return {
        'T': CP.PropsSI('T', 'P', P, 'Q', Q, fluid),
        'P': P,
        'rho': CP.PropsSI('D', 'P', P, 'Q', Q, fluid),
        'h': CP.PropsSI('H', 'P', P, 'Q', Q, fluid),
        's': CP.PropsSI('S', 'P', P, 'Q', Q, fluid),
        'Q': Q,
        'mu': CP.PropsSI('V', 'P', P, 'Q', Q, fluid),
        'k': CP.PropsSI('L', 'P', P, 'Q', Q, fluid),
        'cp': CP.PropsSI('Cpmass', 'P', P, 'Q', Q, fluid),
        'a': CP.PropsSI('A', 'P', P, 'Q', Q, fluid),
    }


def render_orifice_models(fluid_name: str, T_crit: float, P_crit: float):
    """Render two-phase orifice flow model comparison."""
    st.markdown("### Two-Phase Orifice Flow Models")
    
    st.markdown("""
    Compare different models for predicting mass flow through an orifice with 
    potential flashing (two-phase) flow. Critical for N2O injector design.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Orifice Parameters")
        
        d_orifice = st.number_input("Orifice Diameter [mm]", value=2.0, min_value=0.1)
        cd = st.number_input("Discharge Coefficient", value=0.65, min_value=0.3, max_value=1.0)
        
        st.markdown("#### Upstream Conditions")
        P_tank = st.number_input("Tank Pressure [bar]", value=50.0, min_value=1.0)
        T_tank = st.number_input("Tank Temperature [K]", value=293.15, min_value=200.0)
        
        st.markdown("#### Downstream Range")
        P_back_min = st.number_input("Min Back Pressure [bar]", value=1.0)
        P_back_max = st.number_input("Max Back Pressure [bar]", value=float(P_tank * 0.95))
    
    with col2:
        # Calculate flow curves for different models
        A_orifice = np.pi * (d_orifice / 1000 / 2) ** 2
        
        try:
            # Get saturation pressure at tank temp
            P_sat = CP.PropsSI('P', 'T', T_tank, 'Q', 0, fluid_name)
            rho_l = CP.PropsSI('D', 'T', T_tank, 'Q', 0, fluid_name)
            h_l = CP.PropsSI('H', 'T', T_tank, 'Q', 0, fluid_name)
            s_l = CP.PropsSI('S', 'T', T_tank, 'Q', 0, fluid_name)
            rho_l_sat, rho_v_sat = rho_l, CP.PropsSI('D', 'T', T_tank, 'Q', 1, fluid_name)
            
            # Get inlet state
            rho_in = CP.PropsSI('D', 'P', P_tank * 1e5, 'T', T_tank, fluid_name)
            h_in = CP.PropsSI('H', 'P', P_tank * 1e5, 'T', T_tank, fluid_name)
            s_in = CP.PropsSI('S', 'P', P_tank * 1e5, 'T', T_tank, fluid_name)
            
            P_back_range = np.linspace(P_back_max, P_back_min, 100) * 1e5
            
            m_spi = []
            m_hem = []
            m_dyer = []
            
            for P_back in P_back_range:
                # SPI Model
                dp = P_tank * 1e5 - P_back
                if dp > 0:
                    m_spi.append(cd * A_orifice * np.sqrt(2 * rho_in * dp))
                else:
                    m_spi.append(0)
                
                # HEM Model
                try:
                    rho_out = CP.PropsSI('D', 'P', P_back, 'S', s_in, fluid_name)
                    h_out = CP.PropsSI('H', 'P', P_back, 'S', s_in, fluid_name)
                    dh = h_in - h_out
                    if dh > 0:
                        m_hem.append(cd * A_orifice * rho_out * np.sqrt(2 * dh))
                    else:
                        m_hem.append(0)
                except:
                    m_hem.append(0)
                
                # Dyer Model
                if P_back >= P_sat:
                    m_dyer.append(m_spi[-1])
                else:
                    try:
                        k = np.sqrt((P_tank * 1e5 - P_back) / (P_sat - P_back))
                        m_dyer.append((m_spi[-1] + k * m_hem[-1]) / (1 + k))
                    except:
                        m_dyer.append(m_spi[-1])
            
            # Plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=P_back_range / 1e5,
                y=np.array(m_spi) * 1000,
                mode='lines',
                name='SPI (Incompressible)',
                line=dict(color='gray', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=P_back_range / 1e5,
                y=np.array(m_hem) * 1000,
                mode='lines',
                name='HEM (Equilibrium)',
                line=dict(color='green', dash='dashdot')
            ))
            
            fig.add_trace(go.Scatter(
                x=P_back_range / 1e5,
                y=np.array(m_dyer) * 1000,
                mode='lines',
                name='Dyer (NHNE)',
                line=dict(color='blue', width=3)
            ))
            
            # Mark saturation pressure
            fig.add_vline(x=P_sat/1e5, line_dash="dot", line_color="red",
                         annotation_text=f"P_sat = {P_sat/1e5:.1f} bar")
            
            # Find and mark choke point
            idx_max_dyer = np.argmax(m_dyer)
            fig.add_trace(go.Scatter(
                x=[P_back_range[idx_max_dyer] / 1e5],
                y=[m_dyer[idx_max_dyer] * 1000],
                mode='markers',
                name='Choke Point',
                marker=dict(size=12, color='red', symbol='star')
            ))
            
            fig.update_layout(
                title="Mass Flow vs Back Pressure",
                xaxis_title="Back Pressure [bar]",
                yaxis_title="Mass Flow [g/s]",
                height=450,
                xaxis=dict(autorange='reversed'),
                legend=dict(x=0.02, y=0.98)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            st.markdown("#### Critical Flow Conditions")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Max Flow (Dyer)", f"{max(m_dyer)*1000:.2f} g/s")
            with col_b:
                st.metric("Critical Pressure", f"{P_back_range[idx_max_dyer]/1e5:.1f} bar")
            with col_c:
                st.metric("Saturation Pressure", f"{P_sat/1e5:.2f} bar")
            
        except Exception as e:
            st.error(f"Calculation error: {e}")
            import traceback
            st.code(traceback.format_exc())
