"""
Engine Design Page - Main design workflow
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

# Import core modules
from rocket_engine.core.config import EngineConfig, MATERIAL_CONDUCTIVITY
from rocket_engine.core.results import EngineDesignResult


def render_design_page():
    """Render the engine design page."""
    st.title("🔧 Engine Design")
    
    # Check if we have an existing config
    if st.session_state.engine_config:
        st.info(f"Editing: **{st.session_state.engine_config.engine_name}**")
    
    # Create tabs for organized input
    tab_basic, tab_nozzle, tab_cooling, tab_advanced = st.tabs([
        "📝 Basic Parameters",
        "🎯 Nozzle Design", 
        "❄️ Cooling System",
        "⚙️ Advanced Settings"
    ])
    
    # Initialize config dict
    config_dict = {}
    
    # === BASIC PARAMETERS TAB ===
    with tab_basic:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Engine Identification")
            config_dict['engine_name'] = st.text_input(
                "Engine Name",
                value=st.session_state.engine_config.engine_name if st.session_state.engine_config else "New Engine",
                help="Descriptive name for your engine design"
            )
            
            config_dict['designer'] = st.text_input(
                "Designer",
                value=st.session_state.engine_config.designer if st.session_state.engine_config else "",
            )
            
            st.markdown("#### Propellants")
            
            fuel_options = [
                "Ethanol90", "Ethanol80", "Ethanol", "Methane", "RP1", 
                "Propane", "IPA", "C2H6"
            ]
            config_dict['fuel'] = st.selectbox(
                "Fuel",
                fuel_options,
                index=fuel_options.index(st.session_state.engine_config.fuel) 
                    if st.session_state.engine_config and st.session_state.engine_config.fuel in fuel_options 
                    else 0,
                help="Select fuel type"
            )
            
            ox_options = ["N2O", "LOX", "H2O2"]
            config_dict['oxidizer'] = st.selectbox(
                "Oxidizer", 
                ox_options,
                index=0,
                help="Select oxidizer type"
            )
        
        with col2:
            st.markdown("#### Performance Targets")
            
            config_dict['thrust_n'] = st.number_input(
                "Target Thrust [N]",
                min_value=10.0,
                max_value=1000000.0,
                value=st.session_state.engine_config.thrust_n if st.session_state.engine_config else 2000.0,
                step=100.0,
                help="Desired sea-level thrust"
            )
            
            config_dict['pc_bar'] = st.number_input(
                "Chamber Pressure [bar]",
                min_value=5.0,
                max_value=300.0,
                value=st.session_state.engine_config.pc_bar if st.session_state.engine_config else 25.0,
                step=1.0,
                help="Combustion chamber pressure"
            )
            
            config_dict['mr'] = st.number_input(
                "Mixture Ratio (O/F)",
                min_value=0.5,
                max_value=15.0,
                value=st.session_state.engine_config.mr if st.session_state.engine_config else 4.0,
                step=0.1,
                help="Oxidizer to fuel mass ratio"
            )
            
            config_dict['eff_combustion'] = st.slider(
                "Combustion Efficiency",
                min_value=0.80,
                max_value=1.0,
                value=st.session_state.engine_config.eff_combustion if st.session_state.engine_config else 0.95,
                step=0.01,
                help="Expected combustion efficiency (c* efficiency)"
            )
    
    # === NOZZLE DESIGN TAB ===
    with tab_nozzle:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Throat Sizing")
            
            sizing_mode = st.radio(
                "Sizing Mode",
                ["Auto (from Thrust)", "Manual (fixed throat)"],
                help="Choose whether to calculate throat from thrust or specify directly"
            )
            
            if sizing_mode == "Manual (fixed throat)":
                throat_dia_mm = st.number_input(
                    "Throat Diameter [mm]",
                    min_value=1.0,
                    max_value=500.0,
                    value=28.0,
                    step=0.5
                )
                config_dict['throat_diameter'] = throat_dia_mm / 1000.0
            else:
                config_dict['throat_diameter'] = 0.0
                st.info("Throat will be sized to achieve target thrust")
            
            st.markdown("#### Expansion")
            
            expansion_mode = st.radio(
                "Expansion Ratio",
                ["Optimal (for exit pressure)", "Manual"],
                help="Calculate optimal expansion or specify manually"
            )
            
            if expansion_mode == "Manual":
                config_dict['expansion_ratio'] = st.number_input(
                    "Area Ratio (Ae/At)",
                    min_value=1.5,
                    max_value=100.0,
                    value=4.0,
                    step=0.5
                )
            else:
                config_dict['expansion_ratio'] = 0.0
                config_dict['p_exit_bar'] = st.number_input(
                    "Design Exit Pressure [bar]",
                    min_value=0.001,
                    max_value=5.0,
                    value=1.013,
                    step=0.1,
                    help="Ambient pressure for optimal expansion"
                )
        
        with col2:
            st.markdown("#### Chamber Geometry")
            
            config_dict['L_star'] = st.number_input(
                "L* (Characteristic Length) [mm]",
                min_value=500.0,
                max_value=3000.0,
                value=st.session_state.engine_config.L_star if st.session_state.engine_config else 1100.0,
                step=50.0,
                help="Chamber volume / throat area ratio"
            )
            
            config_dict['contraction_ratio'] = st.number_input(
                "Contraction Ratio (Ac/At)",
                min_value=2.0,
                max_value=20.0,
                value=st.session_state.engine_config.contraction_ratio if st.session_state.engine_config else 10.0,
                step=0.5,
                help="Chamber to throat area ratio"
            )
            
            st.markdown("#### Nozzle Profile")
            
            config_dict['bell_fraction'] = st.slider(
                "Bell Fraction (%Rao)",
                min_value=0.6,
                max_value=1.0,
                value=st.session_state.engine_config.bell_fraction if st.session_state.engine_config else 0.8,
                step=0.05,
                help="Nozzle length as fraction of 15° conical"
            )
            
            config_dict['theta_convergent'] = st.number_input(
                "Convergent Half-Angle [deg]",
                min_value=15.0,
                max_value=60.0,
                value=st.session_state.engine_config.theta_convergent if st.session_state.engine_config else 30.0,
                step=5.0
            )
    
    # === COOLING SYSTEM TAB ===
    with tab_cooling:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Coolant Selection")
            
            coolant_options = [
                ("N2O (Nitrous Oxide)", "REFPROP::NitrousOxide"),
                ("Ethanol", "REFPROP::Ethanol"),
                ("Water", "Water"),
                ("LOX (Oxygen)", "REFPROP::Oxygen"),
            ]
            
            coolant_names = [c[0] for c in coolant_options]
            coolant_selected = st.selectbox("Coolant Fluid", coolant_names)
            config_dict['coolant_name'] = next(c[1] for c in coolant_options if c[0] == coolant_selected)
            
            config_dict['cooling_mode'] = st.selectbox(
                "Flow Direction",
                ["counter-flow", "co-flow"],
                help="Counter-flow: inlet at nozzle exit, Co-flow: inlet at injector"
            )
            
            config_dict['coolant_mass_fraction'] = st.slider(
                "Coolant Mass Fraction",
                min_value=0.5,
                max_value=1.0,
                value=1.0,
                step=0.05,
                help="Fraction of oxidizer (or fuel) used for cooling"
            )
            
            st.markdown("#### Coolant Inlet Conditions")
            
            config_dict['coolant_p_in_bar'] = st.number_input(
                "Inlet Pressure [bar]",
                min_value=10.0,
                max_value=200.0,
                value=st.session_state.engine_config.coolant_p_in_bar if st.session_state.engine_config else 50.0,
                step=5.0,
                help="Must be higher than chamber pressure!"
            )
            
            config_dict['coolant_t_in_k'] = st.number_input(
                "Inlet Temperature [K]",
                min_value=100.0,
                max_value=400.0,
                value=st.session_state.engine_config.coolant_t_in_k if st.session_state.engine_config else 290.0,
                step=5.0
            )
        
        with col2:
            st.markdown("#### Channel Geometry (at Throat)")
            
            config_dict['channel_width_throat'] = st.number_input(
                "Channel Width [mm]",
                min_value=0.3,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Width of cooling channels at throat"
            ) / 1000.0
            
            config_dict['channel_height'] = st.number_input(
                "Channel Height [mm]",
                min_value=0.3,
                max_value=5.0,
                value=0.75,
                step=0.1
            ) / 1000.0
            
            config_dict['rib_width_throat'] = st.number_input(
                "Rib Width [mm]",
                min_value=0.3,
                max_value=3.0,
                value=0.6,
                step=0.1,
                help="Width of ribs between channels"
            ) / 1000.0
            
            st.markdown("#### Wall Properties")
            
            config_dict['wall_thickness'] = st.number_input(
                "Wall Thickness [mm]",
                min_value=0.3,
                max_value=3.0,
                value=0.5,
                step=0.1
            ) / 1000.0
            
            material_options = list(MATERIAL_CONDUCTIVITY.keys())
            config_dict['wall_material'] = st.selectbox(
                "Wall Material",
                material_options,
                index=1,  # Default to Inconel 718
                help="Material affects thermal conductivity"
            )
            config_dict['wall_conductivity'] = MATERIAL_CONDUCTIVITY[config_dict['wall_material']]
            
            config_dict['wall_roughness'] = st.number_input(
                "Surface Roughness [µm]",
                min_value=1.0,
                max_value=100.0,
                value=20.0,
                step=5.0
            ) * 1e-6
    
    # === ADVANCED SETTINGS TAB ===
    with tab_advanced:
        st.markdown("#### Injector Preliminary Sizing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dp_mode = st.radio(
                "Injector Pressure Drop",
                ["Auto (15% of Pc)", "Manual"]
            )
            
            if dp_mode == "Manual":
                config_dict['injector_dp_bar'] = st.number_input(
                    "Injector ΔP [bar]",
                    min_value=1.0,
                    max_value=50.0,
                    value=5.0,
                    step=0.5
                )
            else:
                config_dict['injector_dp_bar'] = 0.0  # Will be auto-calculated
        
        with col2:
            st.markdown("#### Solver Settings")
            
            st.number_input(
                "Cooling Stations",
                min_value=50,
                max_value=500,
                value=100,
                step=50,
                help="Number of axial stations for cooling analysis"
            )
    
    st.markdown("---")
    
    # === VALIDATION AND RUN ===
    col_val, col_run = st.columns([2, 1])
    
    with col_val:
        # Create config and validate
        try:
            config = EngineConfig(**config_dict)
            validation = config.validate()
            
            if validation.errors:
                for err in validation.errors:
                    st.error(f"❌ {err}")
            
            if validation.warnings:
                with st.expander("⚠️ Warnings", expanded=True):
                    for warn in validation.warnings:
                        st.warning(warn)
            
            if validation.is_valid:
                st.success("✅ Configuration is valid")
                st.session_state.engine_config = config
                
        except Exception as e:
            st.error(f"Configuration error: {e}")
            config = None
            validation = None
    
    with col_run:
        st.markdown("#### Actions")
        
        run_design = st.button(
            "🚀 Run Design Analysis",
            type="primary",
            disabled=not (validation and validation.is_valid),
            use_container_width=True
        )
        
        if run_design:
            run_engine_design(config)
        
        st.button(
            "💾 Save Configuration",
            use_container_width=True,
            disabled=config is None
        )


def run_engine_design(config: EngineConfig):
    """Execute the engine design analysis."""
    st.markdown("---")
    st.markdown("### 🔄 Running Analysis...")
    
    progress = st.progress(0)
    status = st.empty()
    
    try:
        # Step 1: Combustion Analysis
        status.text("Running combustion analysis (CEA)...")
        progress.progress(10)
        
        # Import actual solver (mocked here for demonstration)
        # from rocket_engine.solvers.engine_solver import EngineSolver
        # solver = EngineSolver(config)
        # result = solver.design()
        
        # For demonstration, create mock result
        import time
        time.sleep(0.5)
        progress.progress(30)
        
        status.text("Generating nozzle geometry...")
        time.sleep(0.5)
        progress.progress(50)
        
        status.text("Computing gas dynamics...")
        time.sleep(0.5)
        progress.progress(70)
        
        status.text("Solving regenerative cooling...")
        time.sleep(0.5)
        progress.progress(90)
        
        # Create mock result for UI demonstration
        result = create_mock_result(config)
        
        progress.progress(100)
        status.text("Analysis complete!")
        
        # Store result
        st.session_state.design_result = result
        
        # Display results
        display_design_results(result)
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        import traceback
        st.code(traceback.format_exc())


def create_mock_result(config: EngineConfig) -> EngineDesignResult:
    """Create a mock result for UI demonstration."""
    from rocket_engine.core.results import (
        EngineDesignResult, CombustionResult, CoolingResult
    )
    
    # Mock combustion
    combustion = CombustionResult(
        pc_bar=config.pc_bar,
        mr=config.mr,
        cstar=1520.0,
        isp_vac=280.0,
        isp_opt=245.0,
        T_combustion=3150.0,
        gamma=1.22,
        mw=26.5,
        mach_exit=2.8
    )
    
    # Mock arrays
    n_stations = 100
    x = np.linspace(-80, 50, n_stations)
    
    # Mock cooling
    cooling = CoolingResult(
        T_coolant=np.linspace(290, 450, n_stations),
        P_coolant=np.linspace(50e5, 40e5, n_stations),
        T_wall_hot=np.linspace(600, 1200, n_stations // 2).tolist() + 
                   np.linspace(1200, 800, n_stations - n_stations // 2).tolist(),
        T_wall_cold=np.linspace(400, 800, n_stations // 2).tolist() + 
                    np.linspace(800, 600, n_stations - n_stations // 2).tolist(),
        q_flux=np.concatenate([np.linspace(1e6, 15e6, n_stations // 2),
                               np.linspace(15e6, 5e6, n_stations - n_stations // 2)]),
        velocity=np.linspace(5, 25, n_stations),
        density=np.linspace(750, 400, n_stations),
        quality=np.full(n_stations, -1.0)
    )
    
    result = EngineDesignResult(
        run_type="design",
        pc_bar=config.pc_bar,
        mr=config.mr,
        isp_vac=combustion.isp_vac * config.eff_combustion,
        isp_sea=combustion.isp_opt * config.eff_combustion,
        thrust_vac=config.thrust_n * 1.15,
        thrust_sea=config.thrust_n,
        massflow_total=config.thrust_n / (combustion.isp_opt * config.eff_combustion * 9.81),
        dt_mm=28.0,
        de_mm=56.0,
        length_mm=130.0,
        expansion_ratio=4.0,
        combustion=combustion,
        cooling=cooling,
        mach_numbers=np.concatenate([np.linspace(0.1, 1.0, n_stations // 2),
                                     np.linspace(1.0, 2.8, n_stations - n_stations // 2)]),
        T_gas_recovery=np.full(n_stations, 2800.0),
        h_gas=np.concatenate([np.linspace(1000, 8000, n_stations // 2),
                              np.linspace(8000, 2000, n_stations - n_stations // 2)])
    )
    
    return result


def display_design_results(result: EngineDesignResult):
    """Display the design analysis results."""
    st.markdown("---")
    st.markdown("## 📊 Design Results")
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Sea-Level Thrust",
            f"{result.thrust_sea:.0f} N",
            f"{(result.thrust_vac - result.thrust_sea):.0f} N (vac bonus)"
        )
    
    with col2:
        st.metric(
            "Specific Impulse",
            f"{result.isp_sea:.1f} s",
            f"+{(result.isp_vac - result.isp_sea):.1f} s vacuum"
        )
    
    with col3:
        st.metric(
            "Mass Flow",
            f"{result.massflow_total:.3f} kg/s"
        )
    
    with col4:
        st.metric(
            "Max Wall Temp",
            f"{result.cooling.max_wall_temp:.0f} K" if result.cooling else "N/A",
            delta=None
        )
    
    # Detailed tabs
    tab_perf, tab_geom, tab_thermal, tab_data = st.tabs([
        "📈 Performance",
        "📐 Geometry", 
        "🌡️ Thermal",
        "📁 Data Export"
    ])
    
    with tab_perf:
        render_performance_tab(result)
    
    with tab_geom:
        render_geometry_tab(result)
    
    with tab_thermal:
        render_thermal_tab(result)
    
    with tab_data:
        render_data_export_tab(result)


def render_performance_tab(result: EngineDesignResult):
    """Render performance summary."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Combustion Properties")
        if result.combustion:
            data = {
                "Combustion Temperature": f"{result.combustion.T_combustion:.0f} K",
                "C* (characteristic velocity)": f"{result.combustion.cstar:.1f} m/s",
                "Gamma (γ)": f"{result.combustion.gamma:.3f}",
                "Molecular Weight": f"{result.combustion.mw:.2f} kg/kmol",
                "Exit Mach Number": f"{result.combustion.mach_exit:.2f}",
            }
            for key, val in data.items():
                st.text(f"{key}: {val}")
    
    with col2:
        st.markdown("#### Operating Point")
        data = {
            "Chamber Pressure": f"{result.pc_bar:.1f} bar",
            "Mixture Ratio (O/F)": f"{result.mr:.2f}",
            "Expansion Ratio": f"{result.expansion_ratio:.2f}",
            "Throat Diameter": f"{result.dt_mm:.2f} mm",
            "Exit Diameter": f"{result.de_mm:.2f} mm",
        }
        for key, val in data.items():
            st.text(f"{key}: {val}")


def render_geometry_tab(result: EngineDesignResult):
    """Render geometry visualization."""
    st.markdown("#### Nozzle Profile")
    
    # Create nozzle profile plot
    if result.mach_numbers is not None:
        n = len(result.mach_numbers)
        x = np.linspace(-80, 50, n)
        
        # Mock radius profile
        y = np.concatenate([
            np.full(30, 30),  # Chamber
            np.linspace(30, 14, 20),  # Convergent
            np.linspace(14, 28, 50)  # Divergent
        ])
        
        fig = go.Figure()
        
        # Upper contour
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='#1a1a2e', width=3),
            name='Chamber Wall'
        ))
        
        # Lower contour (mirror)
        fig.add_trace(go.Scatter(
            x=x, y=-y,
            mode='lines',
            line=dict(color='#1a1a2e', width=3),
            showlegend=False
        ))
        
        # Throat line
        fig.add_vline(x=0, line_dash="dash", line_color="red", 
                      annotation_text="Throat")
        
        fig.update_layout(
            title="Nozzle Contour",
            xaxis_title="Axial Position [mm]",
            yaxis_title="Radius [mm]",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_thermal_tab(result: EngineDesignResult):
    """Render thermal analysis results."""
    if result.cooling is None:
        st.warning("No cooling data available")
        return
    
    cool = result.cooling
    n = len(cool.T_coolant)
    x = np.linspace(-80, 50, n)
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Temperature Profile", 
            "Heat Flux",
            "Coolant Pressure", 
            "Coolant Properties"
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Temperature profiles
    fig.add_trace(
        go.Scatter(x=x, y=cool.T_wall_hot, name="Wall (Hot)", 
                   line=dict(color='red')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=cool.T_coolant, name="Coolant", 
                   line=dict(color='blue')),
        row=1, col=1
    )
    
    # Heat flux
    fig.add_trace(
        go.Scatter(x=x, y=np.array(cool.q_flux) / 1e6, name="Heat Flux",
                   line=dict(color='orange'), fill='tozeroy'),
        row=1, col=2
    )
    
    # Pressure
    fig.add_trace(
        go.Scatter(x=x, y=np.array(cool.P_coolant) / 1e5, name="Pressure",
                   line=dict(color='cyan')),
        row=2, col=1
    )
    
    # Velocity and Density
    fig.add_trace(
        go.Scatter(x=x, y=cool.velocity, name="Velocity [m/s]",
                   line=dict(color='green')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Axial Position [mm]")
    fig.update_yaxes(title_text="Temperature [K]", row=1, col=1)
    fig.update_yaxes(title_text="Heat Flux [MW/m²]", row=1, col=2)
    fig.update_yaxes(title_text="Pressure [bar]", row=2, col=1)
    fig.update_yaxes(title_text="Velocity [m/s]", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Max Wall Temp", f"{cool.max_wall_temp:.0f} K")
    with col2:
        st.metric("Max Heat Flux", f"{cool.max_heat_flux/1e6:.1f} MW/m²")
    with col3:
        st.metric("Pressure Drop", f"{cool.pressure_drop/1e5:.1f} bar")


def render_data_export_tab(result: EngineDesignResult):
    """Render data export options."""
    st.markdown("#### Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Summary Report**")
        if st.button("📄 Generate PDF Report"):
            st.info("PDF generation would be implemented here")
        
        st.markdown("**Configuration**")
        if st.button("💾 Download YAML Config"):
            st.info("YAML export would be implemented here")
    
    with col2:
        st.markdown("**Data Tables**")
        if st.button("📊 Download CSV (Profile Data)"):
            st.info("CSV export would be implemented here")
        
        st.markdown("**CAD Export**")
        if st.button("📐 Download DXF (Contour)"):
            st.info("DXF export would be implemented here")
    
    # Show summary table
    st.markdown("---")
    st.markdown("#### Summary Data")
    
    summary = result.summary_dict()
    import pandas as pd
    df = pd.DataFrame(list(summary.items()), columns=["Parameter", "Value"])
    st.dataframe(df, use_container_width=True, hide_index=True)
