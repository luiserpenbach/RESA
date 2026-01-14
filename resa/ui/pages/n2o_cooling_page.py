"""
RESA N2O Cooling Analysis UI Page
==================================

Streamlit page for detailed two-phase N2O regenerative cooling analysis.
Place in: rocket_engine/ui/pages/n2o_cooling_page.py

Add to app.py navigation:
    pages = {
        ...
        '❄️ N2O Cooling': 'n2o_cooling',
    }
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import cooling module
try:
    from rocket_engine.physics.cooling_n2o import (
        N2OCoolingSolver,
        CoolingChannelGenerator,
        CoolingChannelGeometry,
        N2OConstants,
        FlowRegime,
        plot_cooling_results,
        get_n2o_state,
        get_saturation_properties
    )

    COOLING_AVAILABLE = True
except ImportError:
    COOLING_AVAILABLE = False


def render_n2o_cooling_page():
    """Main render function for N2O cooling analysis page."""

    st.title("❄️ N2O Two-Phase Cooling Analysis")

    st.markdown("""
    Detailed regenerative cooling analysis with proper two-phase flow physics for N2O coolant.

    **Features:**
    - Phase-aware heat transfer (subcooled, boiling, supercritical)
    - Critical Heat Flux (CHF) safety margin tracking
    - Two-phase pressure drop (Lockhart-Martinelli)
    - Flow regime identification
    """)

    if not COOLING_AVAILABLE:
        st.error("N2O cooling module not found. Please install cooling_n2o.py in rocket_engine/physics/")
        return

    # Sidebar configuration
    with st.sidebar:
        st.header("N2O Properties Reference")
        st.metric("Critical Temperature", f"{N2OConstants.T_CRIT - 273.15:.1f}°C")
        st.metric("Critical Pressure", f"{N2OConstants.P_CRIT / 1e5:.1f} bar")

        st.divider()
        st.header("Analysis Mode")
        mode = st.radio(
            "Select mode",
            ["Quick Analysis", "Full Engine", "Parametric Study"]
        )

    if mode == "Quick Analysis":
        render_quick_analysis()
    elif mode == "Full Engine":
        render_full_engine_analysis()
    else:
        render_parametric_study()


def render_quick_analysis():
    """Quick single-point analysis for design iteration."""

    st.header("Quick Analysis")
    st.markdown("Analyze cooling at a single operating point.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Operating Conditions")
        P_inlet = st.number_input("Inlet Pressure [bar]", 30.0, 150.0, 50.0, 5.0)
        T_inlet = st.number_input("Inlet Temperature [K]", 250.0, 310.0, 280.0, 5.0)
        m_dot = st.number_input("Mass Flow Rate [kg/s]", 0.1, 5.0, 0.5, 0.1)
        q_flux_peak = st.number_input("Peak Heat Flux [MW/m²]", 1.0, 10.0, 4.0, 0.5)

    with col2:
        st.subheader("Channel Geometry")
        n_channels = st.number_input("Number of Channels", 20, 100, 40, 5)
        width_throat = st.number_input("Width at Throat [mm]", 0.5, 3.0, 1.0, 0.1)
        ar_throat = st.number_input("Aspect Ratio at Throat", 1.0, 5.0, 3.0, 0.5)
        r_throat = st.number_input("Throat Radius [mm]", 10.0, 50.0, 20.0, 5.0)

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Running cooling analysis..."):
            result = run_quick_analysis(
                P_inlet * 1e5, T_inlet, m_dot, q_flux_peak * 1e6,
                n_channels, width_throat * 1e-3, ar_throat, r_throat * 1e-3
            )

        display_results(result)


def render_full_engine_analysis():
    """Full engine cooling analysis with contour import."""

    st.header("Full Engine Analysis")

    # Check for engine config in session state
    if 'engine_config' in st.session_state and st.session_state.engine_config:
        config = st.session_state.engine_config
        st.success(f"Using engine: {config.engine_name}")

        # Get contour and heat flux from session state
        if 'contour_points' in st.session_state:
            contour = st.session_state.contour_points
        else:
            st.warning("No contour data. Run nozzle design first.")
            contour = None

        if 'heat_flux_profile' in st.session_state:
            q_flux = st.session_state.heat_flux_profile
        else:
            st.warning("No heat flux data. Run thermal analysis first.")
            q_flux = None
    else:
        st.info("No engine configuration loaded. Using example geometry.")
        config = None
        contour = None
        q_flux = None

    # Allow manual input
    st.subheader("Coolant Inlet Conditions")
    col1, col2, col3 = st.columns(3)

    with col1:
        P_inlet = st.number_input(
            "Pressure [bar]", 30.0, 150.0,
            config.coolant_p_in_bar if config else 50.0, 5.0
        )
    with col2:
        T_inlet = st.number_input(
            "Temperature [K]", 250.0, 310.0,
            config.coolant_t_in_k if config else 280.0, 5.0
        )
    with col3:
        m_dot = st.number_input(
            "Mass Flow [kg/s]", 0.1, 5.0,
            config.m_dot_total * 0.3 if config else 0.5, 0.1
        )

    st.subheader("Channel Design")
    col1, col2 = st.columns(2)

    with col1:
        n_channels = st.slider("Number of Channels", 20, 100, 40)
        design_approach = st.selectbox(
            "Design Approach",
            ["Variable AR (recommended)", "Constant Velocity", "Custom"]
        )

    with col2:
        if design_approach == "Variable AR (recommended)":
            ar_throat = st.slider("AR at Throat", 1.5, 5.0, 3.0, 0.5)
            ar_chamber = st.slider("AR in Chamber", 1.0, 3.0, 1.5, 0.5)
        elif design_approach == "Constant Velocity":
            G_target = st.slider("Target Mass Flux [kg/m²-s]", 1000, 5000, 3000, 100)

    if st.button("Run Full Analysis", type="primary"):
        with st.spinner("Running detailed cooling analysis..."):
            # Build geometry
            if contour is not None and q_flux is not None:
                result = run_full_analysis_from_data(
                    contour, q_flux, P_inlet * 1e5, T_inlet, m_dot, n_channels
                )
            else:
                # Use example geometry
                result = run_example_analysis(
                    P_inlet * 1e5, T_inlet, m_dot, n_channels
                )

        display_results(result)


def render_parametric_study():
    """Parametric study for design optimization."""

    st.header("Parametric Study")
    st.markdown("Explore how design parameters affect cooling performance.")

    study_type = st.selectbox(
        "Study Type",
        ["Pressure Sweep", "Mass Flow Sweep", "Channel Count Sweep", "CHF Margin Map"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fixed Parameters")
        if study_type != "Pressure Sweep":
            P_inlet = st.number_input("Inlet Pressure [bar]", 30.0, 150.0, 50.0)
        if study_type != "Mass Flow Sweep":
            m_dot = st.number_input("Mass Flow [kg/s]", 0.1, 5.0, 0.5)
        T_inlet = st.number_input("Inlet Temperature [K]", 250.0, 310.0, 280.0)
        q_flux_peak = st.number_input("Peak Heat Flux [MW/m²]", 1.0, 10.0, 4.0)

    with col2:
        st.subheader("Sweep Range")
        if study_type == "Pressure Sweep":
            P_min = st.number_input("Min Pressure [bar]", 20.0, 100.0, 30.0)
            P_max = st.number_input("Max Pressure [bar]", 50.0, 150.0, 100.0)
            n_points = st.slider("Number of Points", 5, 20, 10)
        elif study_type == "Mass Flow Sweep":
            m_min = st.number_input("Min Mass Flow [kg/s]", 0.1, 2.0, 0.2)
            m_max = st.number_input("Max Mass Flow [kg/s]", 0.5, 5.0, 1.0)
            n_points = st.slider("Number of Points", 5, 20, 10)
        elif study_type == "Channel Count Sweep":
            n_min = st.number_input("Min Channels", 20, 60, 20)
            n_max = st.number_input("Max Channels", 40, 100, 80)
            n_points = st.slider("Number of Points", 5, 15, 8)

    if st.button("Run Parametric Study", type="primary"):
        with st.spinner("Running parametric study..."):
            if study_type == "Pressure Sweep":
                fig = run_pressure_sweep(
                    np.linspace(P_min, P_max, n_points) * 1e5,
                    T_inlet, m_dot, q_flux_peak * 1e6
                )
            elif study_type == "Mass Flow Sweep":
                fig = run_mass_flow_sweep(
                    P_inlet * 1e5, T_inlet,
                    np.linspace(m_min, m_max, n_points),
                    q_flux_peak * 1e6
                )
            elif study_type == "Channel Count Sweep":
                fig = run_channel_count_sweep(
                    P_inlet * 1e5, T_inlet, m_dot, q_flux_peak * 1e6,
                    list(range(int(n_min), int(n_max) + 1,
                               max(1, (int(n_max) - int(n_min)) // n_points)))
                )
            else:
                fig = run_chf_margin_map(
                    T_inlet, q_flux_peak * 1e6
                )

        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def run_quick_analysis(P_inlet, T_inlet, m_dot, q_flux_peak,
                       n_channels, width_throat, ar_throat, r_throat):
    """Run quick single-point analysis."""

    # Simple contour
    x_throat = 0.05

    def contour(x):
        r_chamber = r_throat * 2
        if x < x_throat:
            return r_chamber - (r_chamber - r_throat) * (x / x_throat) ** 2
        else:
            return r_throat + (x - x_throat) * 0.3

    # Heat flux profile
    def q_flux(x):
        sigma = 0.015
        return q_flux_peak * np.exp(-(x - x_throat) ** 2 / (2 * sigma ** 2)) + 0.5e6

    # Channel geometry
    def width(x):
        dist = abs(x - x_throat)
        if dist < 0.02:
            return width_throat
        return width_throat * 1.5

    def height(x):
        dist = abs(x - x_throat)
        if dist < 0.02:
            return width_throat * ar_throat
        return width_throat * 1.5 * 1.5

    def rib_width(x):
        r = contour(x)
        total = 2 * np.pi * r / n_channels
        return max(total - width(x), 0.3e-3)

    channel_geom = CoolingChannelGeometry(
        width=width, height=height, rib_width=rib_width,
        wall_thickness=0.5e-3
    )

    solver = N2OCoolingSolver(
        contour=contour,
        channel_geom=channel_geom,
        q_flux_profile=q_flux,
        m_dot=m_dot,
        P_inlet=P_inlet,
        T_inlet=T_inlet,
        x_start=0.0,
        x_end=0.1,
        n_stations=100,
        n_channels=n_channels,
        engine_name="Quick Analysis"
    )

    return solver.solve()


def run_example_analysis(P_inlet, T_inlet, m_dot, n_channels):
    """Run analysis with example geometry."""
    return run_quick_analysis(
        P_inlet, T_inlet, m_dot, 4e6,
        n_channels, 1e-3, 3.0, 0.02
    )


def run_full_analysis_from_data(contour_points, q_flux_points,
                                P_inlet, T_inlet, m_dot, n_channels):
    """Run analysis from imported contour and heat flux data."""
    from scipy.interpolate import interp1d

    contour_func = interp1d(contour_points[:, 0], contour_points[:, 1],
                            kind='cubic', fill_value='extrapolate')
    q_flux_func = interp1d(q_flux_points[:, 0], q_flux_points[:, 1],
                           kind='linear', fill_value='extrapolate')

    x_throat_idx = np.argmin(contour_points[:, 1])
    x_throat = contour_points[x_throat_idx, 0]
    r_throat = contour_points[x_throat_idx, 1]

    generator = CoolingChannelGenerator(
        contour=contour_func,
        x_throat=x_throat,
        r_throat=r_throat,
        n_channels=n_channels
    )

    channel_geom = generator.design_variable_ar()

    solver = N2OCoolingSolver(
        contour=contour_func,
        channel_geom=channel_geom,
        q_flux_profile=q_flux_func,
        m_dot=m_dot,
        P_inlet=P_inlet,
        T_inlet=T_inlet,
        x_start=contour_points[0, 0],
        x_end=contour_points[-1, 0],
        n_stations=100,
        n_channels=n_channels,
        engine_name="Full Engine"
    )

    return solver.solve()


# =============================================================================
# PARAMETRIC STUDY FUNCTIONS
# =============================================================================

def run_pressure_sweep(P_values, T_inlet, m_dot, q_flux_peak):
    """Sweep inlet pressure and show results."""

    results = []
    for P in P_values:
        result = run_quick_analysis(
            P, T_inlet, m_dot, q_flux_peak,
            40, 1e-3, 3.0, 0.02
        )
        results.append({
            'P_bar': P / 1e5,
            'T_out': result.T_outlet - 273.15,
            'dP': result.dP_total / 1e5,
            'max_chf_margin': result.min_chf_margin,
            'max_T_wall': result.max_wall_temp - 273.15,
            'max_quality': result.max_quality
        })

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Outlet Temperature", "CHF Margin",
                                        "Pressure Drop", "Max Quality"))

    P_bar = [r['P_bar'] for r in results]

    fig.add_trace(go.Scatter(x=P_bar, y=[r['T_out'] for r in results],
                             mode='lines+markers', name='T_out'), row=1, col=1)
    fig.add_trace(go.Scatter(x=P_bar, y=[r['max_chf_margin'] for r in results],
                             mode='lines+markers', name='CHF margin'), row=1, col=2)
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=1, col=2)

    fig.add_trace(go.Scatter(x=P_bar, y=[r['dP'] for r in results],
                             mode='lines+markers', name='ΔP'), row=2, col=1)
    fig.add_trace(go.Scatter(x=P_bar, y=[r['max_quality'] for r in results],
                             mode='lines+markers', name='Quality'), row=2, col=2)

    # Add critical pressure line
    fig.add_vline(x=N2OConstants.P_CRIT / 1e5, line_dash="dot",
                  annotation_text="P_crit", row=1, col=1)

    fig.update_xaxes(title_text="Inlet Pressure [bar]")
    fig.update_layout(title="Pressure Sweep Results", height=600)

    return fig


def run_mass_flow_sweep(P_inlet, T_inlet, m_dot_values, q_flux_peak):
    """Sweep mass flow rate."""

    results = []
    for m_dot in m_dot_values:
        result = run_quick_analysis(
            P_inlet, T_inlet, m_dot, q_flux_peak,
            40, 1e-3, 3.0, 0.02
        )
        results.append({
            'm_dot': m_dot,
            'T_out': result.T_outlet - 273.15,
            'dP': result.dP_total / 1e5,
            'max_chf_margin': result.min_chf_margin,
            'max_T_wall': result.max_wall_temp - 273.15
        })

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Outlet Temperature", "Max Wall Temperature",
                                        "CHF Margin", "Pressure Drop"))

    m = [r['m_dot'] for r in results]

    fig.add_trace(go.Scatter(x=m, y=[r['T_out'] for r in results],
                             mode='lines+markers'), row=1, col=1)
    fig.add_trace(go.Scatter(x=m, y=[r['max_T_wall'] for r in results],
                             mode='lines+markers'), row=1, col=2)
    fig.add_trace(go.Scatter(x=m, y=[r['max_chf_margin'] for r in results],
                             mode='lines+markers'), row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_trace(go.Scatter(x=m, y=[r['dP'] for r in results],
                             mode='lines+markers'), row=2, col=2)

    fig.update_xaxes(title_text="Mass Flow Rate [kg/s]")
    fig.update_layout(title="Mass Flow Sweep Results", height=600)

    return fig


def run_channel_count_sweep(P_inlet, T_inlet, m_dot, q_flux_peak, n_values):
    """Sweep number of channels."""

    results = []
    for n in n_values:
        result = run_quick_analysis(
            P_inlet, T_inlet, m_dot, q_flux_peak,
            n, 1e-3, 3.0, 0.02
        )
        results.append({
            'n_channels': n,
            'dP': result.dP_total / 1e5,
            'max_chf_margin': result.min_chf_margin,
            'max_T_wall': result.max_wall_temp - 273.15
        })

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Pressure Drop", "CHF Margin", "Max Wall Temp"))

    n = [r['n_channels'] for r in results]

    fig.add_trace(go.Scatter(x=n, y=[r['dP'] for r in results],
                             mode='lines+markers'), row=1, col=1)
    fig.add_trace(go.Scatter(x=n, y=[r['max_chf_margin'] for r in results],
                             mode='lines+markers'), row=1, col=2)
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_trace(go.Scatter(x=n, y=[r['max_T_wall'] for r in results],
                             mode='lines+markers'), row=1, col=3)

    fig.update_xaxes(title_text="Number of Channels")
    fig.update_layout(title="Channel Count Sweep Results", height=400)

    return fig


def run_chf_margin_map(T_inlet, q_flux_peak):
    """Generate CHF margin map for P vs m_dot."""

    P_range = np.linspace(30, 100, 15) * 1e5
    m_range = np.linspace(0.2, 1.0, 12)

    chf_margin = np.zeros((len(P_range), len(m_range)))

    for i, P in enumerate(P_range):
        for j, m_dot in enumerate(m_range):
            try:
                result = run_quick_analysis(
                    P, T_inlet, m_dot, q_flux_peak,
                    40, 1e-3, 3.0, 0.02
                )
                chf_margin[i, j] = result.min_chf_margin
            except:
                chf_margin[i, j] = np.nan

    fig = go.Figure(data=go.Contour(
        x=m_range,
        y=P_range / 1e5,
        z=chf_margin,
        colorscale='RdYlGn_r',
        contours=dict(
            start=0, end=1, size=0.1,
            showlabels=True
        ),
        colorbar=dict(title='q/q_CHF')
    ))

    # Add safe operating line
    fig.add_trace(go.Contour(
        x=m_range, y=P_range / 1e5, z=chf_margin,
        contours=dict(start=0.5, end=0.5, coloring='lines'),
        line=dict(color='red', width=3),
        showscale=False,
        name='Safety Limit'
    ))

    fig.add_hline(y=N2OConstants.P_CRIT / 1e5, line_dash="dot",
                  annotation_text="P_crit")

    fig.update_layout(
        title="CHF Margin Operating Map (Green = Safe, Red = Danger)",
        xaxis_title="Mass Flow Rate [kg/s]",
        yaxis_title="Inlet Pressure [bar]",
        height=500
    )

    return fig


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def display_results(result):
    """Display cooling analysis results."""

    st.divider()
    st.header("Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Outlet Temperature", f"{result.T_outlet - 273.15:.1f}°C",
                  f"+{result.T_outlet - result.stations[0].fluid.T:.1f}°C")
    with col2:
        st.metric("Pressure Drop", f"{result.dP_total / 1e5:.2f} bar")
    with col3:
        color = "normal" if result.min_chf_margin < 0.5 else "inverse"
        st.metric("Max CHF Margin", f"{result.min_chf_margin:.2f}",
                  delta="SAFE" if result.min_chf_margin < 0.5 else "DANGER!",
                  delta_color=color)
    with col4:
        st.metric("Max Wall Temp", f"{result.max_wall_temp - 273.15:.0f}°C")

    # Additional metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Heat Absorbed", f"{result.Q_total / 1000:.1f} kW")
    with col2:
        st.metric("Max Quality", f"{result.max_quality:.3f}")
    with col3:
        st.metric("Outlet Pressure", f"{result.P_outlet / 1e5:.1f} bar")

    # Flow regimes encountered
    st.subheader("Flow Regimes Encountered")
    regime_cols = st.columns(len(result.regimes_encountered))
    for i, regime in enumerate(result.regimes_encountered):
        with regime_cols[i]:
            st.info(regime.value.replace("_", " ").title())

    # Warnings
    if result.warnings:
        with st.expander(f"⚠️ Warnings ({len(result.warnings)})", expanded=True):
            for w in result.warnings[:10]:
                st.warning(w)
            if len(result.warnings) > 10:
                st.caption(f"...and {len(result.warnings) - 10} more")

    if result.errors:
        st.error("Analysis Errors")
        for e in result.errors:
            st.error(e)

    # Plots
    st.subheader("Detailed Profiles")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Temperature", "Heat Transfer", "Pressure", "Flow Regime"
    ])

    x = [s.x * 1000 for s in result.stations]

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=[s.fluid.T - 273.15 for s in result.stations],
            name='Coolant Temperature', line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=x, y=[s.T_wall_cold - 273.15 for s in result.stations],
            name='Wall (Cold Side)', line=dict(color='orange')
        ))
        fig.add_trace(go.Scatter(
            x=x, y=[s.T_wall_hot - 273.15 for s in result.stations],
            name='Wall (Hot Side)', line=dict(color='red')
        ))

        # Add saturation temperature where applicable
        T_sat = []
        for s in result.stations:
            if s.fluid.P < N2OConstants.P_CRIT:
                try:
                    T_sat.append(get_saturation_properties(s.fluid.P)[0] - 273.15)
                except:
                    T_sat.append(None)
            else:
                T_sat.append(None)

        x_sat = [x[i] for i, t in enumerate(T_sat) if t is not None]
        T_sat_valid = [t for t in T_sat if t is not None]
        if T_sat_valid:
            fig.add_trace(go.Scatter(
                x=x_sat, y=T_sat_valid,
                name='T_sat', line=dict(color='green', dash='dash')
            ))

        fig.update_layout(
            title="Temperature Profile",
            xaxis_title="Position [mm]",
            yaxis_title="Temperature [°C]",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Heat Transfer Coefficient", "CHF Margin"))

        fig.add_trace(go.Scatter(
            x=x, y=[s.h_conv / 1000 for s in result.stations],
            name='h_conv', line=dict(color='green')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x, y=[s.chf_margin for s in result.stations],
            name='q/q_CHF', line=dict(color='red')
        ), row=1, col=2)
        fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                      row=1, col=2, annotation_text="Safety Limit")

        fig.update_xaxes(title_text="Position [mm]")
        fig.update_yaxes(title_text="h [kW/m²-K]", row=1, col=1)
        fig.update_yaxes(title_text="q/q_CHF [-]", row=1, col=2)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Pressure", "Quality"))

        fig.add_trace(go.Scatter(
            x=x, y=[s.fluid.P / 1e5 for s in result.stations],
            name='Pressure', line=dict(color='purple')
        ), row=1, col=1)
        fig.add_hline(y=N2OConstants.P_CRIT / 1e5, line_dash="dot",
                      row=1, col=1, annotation_text="P_crit")

        fig.add_trace(go.Scatter(
            x=x, y=[s.fluid.quality or 0 for s in result.stations],
            name='Quality', line=dict(color='orange')
        ), row=1, col=2)

        fig.update_xaxes(title_text="Position [mm]")
        fig.update_yaxes(title_text="Pressure [bar]", row=1, col=1)
        fig.update_yaxes(title_text="Quality x [-]", row=1, col=2)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # Color map for regimes
        regime_colors = {
            FlowRegime.SUBCOOLED_LIQUID: 'blue',
            FlowRegime.SUBCOOLED_BOILING: 'cyan',
            FlowRegime.SATURATED_BOILING: 'green',
            FlowRegime.ANNULAR_FLOW: 'lime',
            FlowRegime.MIST_FLOW: 'yellow',
            FlowRegime.CHF_RISK: 'orange',
            FlowRegime.POST_CHF: 'red',
            FlowRegime.SUPERHEATED_VAPOR: 'pink',
            FlowRegime.SUPERCRITICAL: 'purple',
            FlowRegime.PSEUDO_CRITICAL: 'magenta',
            FlowRegime.SUPERCRITICAL_GAS_LIKE: 'violet'
        }

        fig = go.Figure()

        # Group by regime for cleaner visualization
        for regime in FlowRegime:
            x_regime = [x[i] for i, s in enumerate(result.stations) if s.flow_regime == regime]
            if x_regime:
                fig.add_trace(go.Scatter(
                    x=x_regime,
                    y=[1] * len(x_regime),
                    mode='markers',
                    marker=dict(
                        color=regime_colors.get(regime, 'gray'),
                        size=12
                    ),
                    name=regime.value.replace("_", " ").title()
                ))

        fig.update_layout(
            title="Flow Regime Along Channel",
            xaxis_title="Position [mm]",
            yaxis=dict(showticklabels=False),
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feasibility assessment
    st.divider()
    if result.is_feasible:
        st.success("✅ Design appears feasible within safety margins")
    else:
        st.error("❌ Design has safety concerns - review warnings and consider modifications")


# Entry point for RESA integration
if __name__ == "__main__":
    st.set_page_config(page_title="N2O Cooling Analysis", layout="wide")
    render_n2o_cooling_page()