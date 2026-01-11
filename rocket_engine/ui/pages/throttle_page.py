"""
Throttle Analysis Page - Engine throttling and operating envelope
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def render_throttle_page():
    """Render the throttle analysis page."""
    st.title("⚡ Throttle Analysis")
    
    # Check for existing design
    if not st.session_state.get('engine_config'):
        st.warning("⚠️ Please complete an engine design first in the Design page.")
        if st.button("Go to Design Page"):
            st.session_state.current_page = 'design'
            st.rerun()
        return
    
    config = st.session_state.engine_config
    
    st.markdown(f"""
    Analyze throttling behavior for **{config.engine_name}**.
    
    This tool simulates the engine response when oxidizer flow is reduced 
    while fuel flow remains controlled (fixed venturi or variable valve).
    """)
    
    st.markdown("---")
    
    # Throttle configuration
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Throttle Configuration")
        
        throttle_mode = st.radio(
            "Throttle Strategy",
            ["Oxidizer Only (Fixed Fuel)", "Both Propellants", "Oxidizer Only (Passive Fuel)"],
            help="""
            - **Fixed Fuel**: Fuel through cavitating venturi (constant mdot)
            - **Both**: Both propellants throttled proportionally
            - **Passive Fuel**: Fuel through orifice (varies with Pc)
            """
        )
        
        st.markdown("### Throttle Range")
        
        throttle_max = st.slider("Maximum Throttle [%]", 50, 100, 100)
        throttle_min = st.slider("Minimum Throttle [%]", 10, 80, 40)
        n_points = st.slider("Analysis Points", 5, 30, 15)
        
        st.markdown("### Operating Limits")
        
        col_lim1, col_lim2 = st.columns(2)
        with col_lim1:
            pc_min_limit = st.number_input("Min Pc [bar]", value=8.0, min_value=1.0)
            mr_min_limit = st.number_input("Min MR", value=2.0, min_value=0.5)
        with col_lim2:
            pc_max_limit = st.number_input("Max Pc [bar]", value=35.0)
            mr_max_limit = st.number_input("Max MR", value=6.0)
        
        run_analysis = st.button("🔄 Run Throttle Analysis", type="primary", use_container_width=True)
    
    with col2:
        if run_analysis:
            with st.spinner("Computing throttle curve..."):
                results = compute_throttle_curve(
                    config, 
                    throttle_mode,
                    throttle_min, 
                    throttle_max, 
                    n_points
                )
                
                st.session_state.throttle_results = results
        
        if st.session_state.get('throttle_results'):
            display_throttle_results(
                st.session_state.throttle_results,
                config,
                (pc_min_limit, pc_max_limit),
                (mr_min_limit, mr_max_limit)
            )
        else:
            st.info("Click 'Run Throttle Analysis' to generate results")


def compute_throttle_curve(config, mode: str, min_pct: float, max_pct: float, n_points: int):
    """Compute throttle curve data."""
    
    # For demonstration, generate synthetic but physically reasonable data
    throttle_pcts = np.linspace(max_pct, min_pct, n_points)
    
    # Design point values
    pc_design = config.pc_bar
    mr_design = config.mr
    thrust_design = config.thrust_n
    
    results = []
    
    for pct in throttle_pcts:
        frac = pct / 100.0
        
        if "Fixed Fuel" in mode:
            # Oxidizer throttled, fuel constant
            # MR drops as ox is reduced
            # Pc drops roughly with total flow
            mr = mr_design * frac
            mr = max(mr, 1.5)  # Physical lower limit
            
            # Total flow = mdot_ox + mdot_fuel
            # With fixed fuel: mdot_total = mdot_ox_design * frac + mdot_fuel_design
            fuel_frac_design = 1 / (1 + mr_design)
            ox_frac_design = mr_design / (1 + mr_design)
            
            mdot_total_frac = ox_frac_design * frac + fuel_frac_design
            
            # Pc roughly proportional to total flow (c* varies slightly with MR)
            cstar_correction = 1.0 - 0.05 * abs(mr - mr_design) / mr_design
            pc = pc_design * mdot_total_frac * cstar_correction
            
            # Thrust scales with Pc and slight Isp variation
            isp_correction = 1.0 - 0.03 * abs(mr - mr_design) / mr_design
            thrust = thrust_design * (pc / pc_design) * isp_correction
            
        elif "Both" in mode:
            # Both throttled proportionally - MR stays constant
            mr = mr_design
            pc = pc_design * frac
            thrust = thrust_design * frac
            
        else:  # Passive fuel
            # Fuel flow varies with sqrt(P_tank - Pc)
            # More complex coupling
            mr = mr_design * (0.7 + 0.3 * frac)  # Simplified
            pc = pc_design * frac ** 0.8
            thrust = thrust_design * frac ** 0.9
        
        # Wall temperature decreases with heat flux (lower at low throttle)
        # But may increase if MR goes very lean (less film cooling)
        T_wall_base = 1100  # K at design
        T_wall = T_wall_base * (0.6 + 0.4 * (pc / pc_design)) * (1.0 + 0.1 * abs(mr - 3.5) / 3.5)
        
        results.append({
            'throttle_pct': pct,
            'pc_bar': pc,
            'mr': mr,
            'thrust_n': thrust,
            'isp_s': thrust / (thrust_design / 250),  # Rough Isp
            'T_wall_k': T_wall
        })
    
    return pd.DataFrame(results)


def display_throttle_results(df: pd.DataFrame, config, pc_limits: tuple, mr_limits: tuple):
    """Display throttle analysis results."""
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Thrust vs Throttle", 
            "Chamber Pressure",
            "Mixture Ratio",
            "Operating Envelope (Pc vs MR)"
        ],
        specs=[
            [{}, {}],
            [{}, {}]
        ]
    )
    
    # Thrust curve
    fig.add_trace(
        go.Scatter(
            x=df['throttle_pct'],
            y=df['thrust_n'],
            mode='lines+markers',
            name='Thrust',
            line=dict(color='#e94560', width=3)
        ),
        row=1, col=1
    )
    
    # Pc curve
    fig.add_trace(
        go.Scatter(
            x=df['throttle_pct'],
            y=df['pc_bar'],
            mode='lines+markers',
            name='Chamber Pressure',
            line=dict(color='#4db6ac', width=3)
        ),
        row=1, col=2
    )
    
    # MR curve  
    fig.add_trace(
        go.Scatter(
            x=df['throttle_pct'],
            y=df['mr'],
            mode='lines+markers',
            name='Mixture Ratio',
            line=dict(color='#ffc107', width=3)
        ),
        row=2, col=1
    )
    
    # Operating envelope (Pc vs MR scatter)
    fig.add_trace(
        go.Scatter(
            x=df['mr'],
            y=df['pc_bar'],
            mode='lines+markers',
            name='Operating Path',
            line=dict(color='#2196f3', width=2),
            marker=dict(size=8, color=df['throttle_pct'], colorscale='Viridis',
                       showscale=True, colorbar=dict(title='Throttle %', x=1.15))
        ),
        row=2, col=2
    )
    
    # Add operating envelope box
    fig.add_shape(
        type="rect",
        x0=mr_limits[0], x1=mr_limits[1],
        y0=pc_limits[0], y1=pc_limits[1],
        line=dict(color="green", width=2, dash="dash"),
        fillcolor="green",
        opacity=0.1,
        row=2, col=2
    )
    
    # Mark design point
    fig.add_trace(
        go.Scatter(
            x=[config.mr],
            y=[config.pc_bar],
            mode='markers',
            name='Design Point',
            marker=dict(size=15, color='red', symbol='star')
        ),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Throttle [%]", row=1, col=1)
    fig.update_xaxes(title_text="Throttle [%]", row=1, col=2)
    fig.update_xaxes(title_text="Throttle [%]", row=2, col=1)
    fig.update_xaxes(title_text="Mixture Ratio (O/F)", row=2, col=2)
    
    fig.update_yaxes(title_text="Thrust [N]", row=1, col=1)
    fig.update_yaxes(title_text="Pressure [bar]", row=1, col=2)
    fig.update_yaxes(title_text="MR", row=2, col=1)
    fig.update_yaxes(title_text="Pc [bar]", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    st.markdown("### Throttle Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        throttle_ratio = df['thrust_n'].max() / df['thrust_n'].min()
        st.metric("Throttle Ratio", f"{throttle_ratio:.1f}:1")
    
    with col2:
        st.metric("Min Thrust", f"{df['thrust_n'].min():.0f} N")
    
    with col3:
        st.metric("MR Range", f"{df['mr'].min():.2f} - {df['mr'].max():.2f}")
    
    with col4:
        st.metric("Pc Range", f"{df['pc_bar'].min():.1f} - {df['pc_bar'].max():.1f} bar")
    
    # Check for limit violations
    st.markdown("### Limit Check")
    
    violations = []
    
    if df['pc_bar'].min() < pc_limits[0]:
        violations.append(f"⚠️ Pc drops below {pc_limits[0]} bar at low throttle")
    
    if df['mr'].min() < mr_limits[0]:
        violations.append(f"⚠️ MR drops below {mr_limits[0]} (very fuel-rich)")
    
    if df['mr'].max() > mr_limits[1]:
        violations.append(f"⚠️ MR exceeds {mr_limits[1]} (very ox-rich)")
    
    if violations:
        for v in violations:
            st.warning(v)
    else:
        st.success("✅ All operating points within specified limits")
    
    # Data table
    with st.expander("📊 View Data Table"):
        st.dataframe(
            df.style.format({
                'throttle_pct': '{:.0f}%',
                'pc_bar': '{:.2f}',
                'mr': '{:.2f}',
                'thrust_n': '{:.0f}',
                'isp_s': '{:.1f}',
                'T_wall_k': '{:.0f}'
            }),
            use_container_width=True
        )
        
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            "throttle_data.csv",
            "text/csv"
        )
