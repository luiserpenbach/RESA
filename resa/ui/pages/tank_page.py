"""Tank Pressurization Simulation Page for RESA UI."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_tank_page():
    """Render the tank pressurization simulation page."""
    st.title("Tank Pressurization Simulation")

    tab1, tab2, tab3 = st.tabs(["Configuration", "Simulation", "Results"])

    with tab1:
        render_config_tab()

    with tab2:
        render_simulation_tab()

    with tab3:
        render_results_tab()


def render_config_tab():
    """Render tank configuration inputs."""
    st.subheader("Tank Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Oxidizer Tank (N2O)")
        ox_volume = st.number_input(
            "Tank Volume [L]",
            10.0, 500.0, 50.0,
            key="ox_volume"
        )
        ox_fill = st.slider(
            "Fill Level [%]",
            50, 95, 80,
            key="ox_fill"
        )
        ox_temp = st.number_input(
            "Initial Temperature [K]",
            270.0, 320.0, 293.15,
            key="ox_temp"
        )

        st.markdown("### Oxidizer Flow")
        ox_mdot = st.number_input(
            "Mass Flow Rate [kg/s]",
            0.1, 5.0, 0.6,
            key="ox_mdot"
        )

        ox_self_press = st.checkbox(
            "Self-Pressurizing (Vapor Blowdown)",
            value=True,
            key="ox_self_press"
        )

    with col2:
        st.markdown("### Fuel Tank (Ethanol)")
        fuel_volume = st.number_input(
            "Tank Volume [L]",
            5.0, 200.0, 30.0,
            key="fuel_volume"
        )
        fuel_fill = st.slider(
            "Fill Level [%]",
            50, 95, 90,
            key="fuel_fill"
        )
        fuel_temp = st.number_input(
            "Initial Temperature [K]",
            270.0, 320.0, 293.15,
            key="fuel_temp"
        )

        st.markdown("### Fuel Flow")
        fuel_mdot = st.number_input(
            "Mass Flow Rate [kg/s]",
            0.05, 2.0, 0.12,
            key="fuel_mdot"
        )

    st.markdown("---")
    st.markdown("### Pressurant System")

    col3, col4 = st.columns(2)

    with col3:
        press_type = st.selectbox(
            "Pressurant Gas",
            ["Nitrogen", "Helium"],
            key="press_gas"
        )
        press_p = st.number_input(
            "Supply Pressure [bar]",
            50.0, 400.0, 200.0,
            key="press_p"
        )

    with col4:
        press_temp = st.number_input(
            "Supply Temperature [K]",
            200.0, 350.0, 293.15,
            key="press_temp"
        )
        reg_cv = st.number_input(
            "Regulator Cv",
            0.0001, 0.01, 0.001,
            format="%.4f",
            key="press_cv"
        )

    st.markdown("---")
    st.markdown("### Environmental")

    col5, col6 = st.columns(2)

    with col5:
        ambient_temp = st.number_input(
            "Ambient Temperature [K]",
            250.0, 330.0, 293.15,
            key="ambient_temp"
        )

    with col6:
        htc = st.number_input(
            "Heat Transfer Coeff [W/m2K]",
            1.0, 100.0, 10.0,
            key="htc"
        )

    if st.button("Save Configuration", type="primary"):
        st.session_state.tank_config = {
            "ox_volume": ox_volume / 1000,  # L -> m3
            "ox_fill": ox_fill / 100,
            "ox_temp": ox_temp,
            "ox_mdot": ox_mdot,
            "ox_self_press": ox_self_press,
            "fuel_volume": fuel_volume / 1000,
            "fuel_fill": fuel_fill / 100,
            "fuel_temp": fuel_temp,
            "fuel_mdot": fuel_mdot,
            "press_type": press_type,
            "press_p": press_p * 1e5,  # bar -> Pa
            "press_temp": press_temp,
            "reg_cv": reg_cv,
            "ambient_temp": ambient_temp,
            "htc": htc,
        }
        st.success("Configuration saved!")


def render_simulation_tab():
    """Render simulation controls."""
    st.subheader("Run Simulation")

    if not st.session_state.get('tank_config'):
        st.warning("Configure tanks first in the Configuration tab.")
        return

    config = st.session_state.tank_config

    st.markdown("### Simulation Parameters")

    col1, col2 = st.columns(2)

    with col1:
        burn_time = st.number_input(
            "Burn Duration [s]",
            10.0, 300.0, 60.0,
            key="burn_time"
        )
        dt = st.number_input(
            "Time Step [s]",
            0.01, 1.0, 0.1,
            key="sim_dt"
        )

    with col2:
        sim_ox = st.checkbox("Simulate Oxidizer Tank", value=True)
        sim_fuel = st.checkbox("Simulate Fuel Tank", value=True)

    # Display config summary
    with st.expander("Configuration Summary"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Oxidizer (N2O)**")
            st.write(f"- Volume: {config['ox_volume']*1000:.1f} L")
            st.write(f"- Fill: {config['ox_fill']*100:.0f}%")
            st.write(f"- Flow: {config['ox_mdot']:.2f} kg/s")
            st.write(f"- Self-pressurizing: {config['ox_self_press']}")

        with col2:
            st.markdown("**Fuel (Ethanol)**")
            st.write(f"- Volume: {config['fuel_volume']*1000:.1f} L")
            st.write(f"- Fill: {config['fuel_fill']*100:.0f}%")
            st.write(f"- Flow: {config['fuel_mdot']:.2f} kg/s")

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running tank depletion simulation..."):
            try:
                results = run_tank_simulation(
                    config, burn_time, sim_ox, sim_fuel
                )
                st.session_state.tank_results = results
                st.success("Simulation complete!")

                # Quick summary
                if results.get('ox'):
                    ox = results['ox']
                    st.markdown("**Oxidizer Tank Summary**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Final Pressure", f"{ox['p_final']/1e5:.1f} bar")
                    with col2:
                        st.metric("Final Temp", f"{ox['T_final']:.1f} K")
                    with col3:
                        st.metric("Mass Expelled", f"{ox['mass_used']:.2f} kg")

                if results.get('fuel'):
                    fuel = results['fuel']
                    st.markdown("**Fuel Tank Summary**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Final Pressure", f"{fuel['p_final']/1e5:.1f} bar")
                    with col2:
                        st.metric("Final Temp", f"{fuel['T_final']:.1f} K")
                    with col3:
                        st.metric("Mass Expelled", f"{fuel['mass_used']:.2f} kg")

            except Exception as e:
                st.error(f"Simulation failed: {e}")


def run_tank_simulation(config, burn_time, sim_ox, sim_fuel):
    """Run tank simulation with given configuration."""
    results = {}

    try:
        from resa.addons.tank import (
            TankConfig, PressurantConfig, PropellantConfig,
            TwoPhaseNitrousTank, EthanolTank
        )
    except ImportError:
        # Fallback simulation for demo purposes
        return run_demo_simulation(config, burn_time, sim_ox, sim_fuel)

    if sim_ox:
        # N2O tank (self-pressurizing two-phase)
        tank_config = TankConfig(
            volume=config['ox_volume'],
            initial_liquid_mass=config['ox_volume'] * config['ox_fill'] * 800,  # Approx N2O density
            initial_ullage_pressure=50e5,  # Will equilibrate
            initial_temperature=config['ox_temp'],
            wall_material_properties={},
            ambient_temperature=config['ambient_temp'],
            heat_transfer_coefficient=config['htc']
        )

        press_config = PressurantConfig(
            fluid_name=config['press_type'],
            supply_pressure=config['press_p'],
            supply_temperature=config['press_temp'],
            regulator_flow_coefficient=config['reg_cv']
        )

        prop_config = PropellantConfig(
            fluid_name='NitrousOxide',
            mass_flow_rate=config['ox_mdot'],
            is_self_pressurizing=config['ox_self_press']
        )

        try:
            tank = TwoPhaseNitrousTank(tank_config, press_config, prop_config)
            sol = tank.simulate((0, burn_time))

            results['ox'] = {
                't': sol.t,
                'p': sol.y[0],  # Pressure
                'T': sol.y[1],  # Temperature
                'm_liquid': sol.y[2],  # Liquid mass
                'p_final': sol.y[0, -1],
                'T_final': sol.y[1, -1],
                'mass_used': sol.y[2, 0] - sol.y[2, -1],
            }
        except Exception as e:
            st.warning(f"N2O simulation error: {e}")
            results['ox'] = run_demo_ox(config, burn_time)

    if sim_fuel:
        # Fuel tank (regulated pressurization)
        results['fuel'] = run_demo_fuel(config, burn_time)

    return results


def run_demo_simulation(config, burn_time, sim_ox, sim_fuel):
    """Demo simulation when full physics module unavailable."""
    results = {}

    if sim_ox:
        results['ox'] = run_demo_ox(config, burn_time)

    if sim_fuel:
        results['fuel'] = run_demo_fuel(config, burn_time)

    return results


def run_demo_ox(config, burn_time):
    """Demo N2O tank simulation."""
    t = np.linspace(0, burn_time, 500)

    # Initial conditions (approximate N2O at 20C)
    p0 = 50e5  # ~50 bar vapor pressure
    T0 = config['ox_temp']
    m0 = config['ox_volume'] * config['ox_fill'] * 800  # kg

    # Simple exponential decay model
    m_liquid = m0 - config['ox_mdot'] * t
    m_liquid = np.maximum(m_liquid, 0)

    # Pressure drops as liquid depletes (blowdown)
    depletion = 1 - m_liquid / m0
    p = p0 * (1 - 0.3 * depletion)  # Pressure drops ~30%

    # Temperature drops due to evaporative cooling
    T = T0 - 20 * depletion  # ~20K drop

    return {
        't': t,
        'p': p,
        'T': T,
        'm_liquid': m_liquid,
        'p_final': p[-1],
        'T_final': T[-1],
        'mass_used': m0 - m_liquid[-1],
    }


def run_demo_fuel(config, burn_time):
    """Demo fuel tank simulation."""
    t = np.linspace(0, burn_time, 500)

    # Initial conditions
    p0 = config['press_p'] * 0.7  # Regulated to ~70% supply
    T0 = config['fuel_temp']
    rho_fuel = 789  # Ethanol density kg/m3
    m0 = config['fuel_volume'] * config['fuel_fill'] * rho_fuel

    # Mass decreases linearly
    m_liquid = m0 - config['fuel_mdot'] * t
    m_liquid = np.maximum(m_liquid, 0)

    # Pressure stays relatively constant (regulated)
    p = np.ones_like(t) * p0

    # Small temperature change
    T = T0 * np.ones_like(t)

    return {
        't': t,
        'p': p,
        'T': T,
        'm_liquid': m_liquid,
        'p_final': p[-1],
        'T_final': T[-1],
        'mass_used': m0 - m_liquid[-1],
    }


def render_results_tab():
    """Render simulation results and plots."""
    st.subheader("Simulation Results")

    if not st.session_state.get('tank_results'):
        st.warning("Run simulation first.")
        return

    results = st.session_state.tank_results

    # Create plots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Oxidizer Pressure", "Fuel Pressure",
            "Oxidizer Temperature", "Fuel Temperature",
            "Oxidizer Mass", "Fuel Mass"
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    if results.get('ox'):
        ox = results['ox']
        fig.add_trace(
            go.Scatter(x=ox['t'], y=ox['p']/1e5, name='N2O Pressure',
                       line=dict(color='#2196F3', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=ox['t'], y=ox['T'], name='N2O Temp',
                       line=dict(color='#2196F3', width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=ox['t'], y=ox['m_liquid'], name='N2O Mass',
                       line=dict(color='#2196F3', width=2)),
            row=3, col=1
        )

    if results.get('fuel'):
        fuel = results['fuel']
        fig.add_trace(
            go.Scatter(x=fuel['t'], y=fuel['p']/1e5, name='Fuel Pressure',
                       line=dict(color='#FF9800', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=fuel['t'], y=fuel['T'], name='Fuel Temp',
                       line=dict(color='#FF9800', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=fuel['t'], y=fuel['m_liquid'], name='Fuel Mass',
                       line=dict(color='#FF9800', width=2)),
            row=3, col=2
        )

    # Update axes labels
    fig.update_xaxes(title_text="Time [s]", row=3)
    fig.update_yaxes(title_text="Pressure [bar]", row=1, col=1)
    fig.update_yaxes(title_text="Pressure [bar]", row=1, col=2)
    fig.update_yaxes(title_text="Temperature [K]", row=2, col=1)
    fig.update_yaxes(title_text="Temperature [K]", row=2, col=2)
    fig.update_yaxes(title_text="Mass [kg]", row=3, col=1)
    fig.update_yaxes(title_text="Mass [kg]", row=3, col=2)

    fig.update_layout(
        height=700,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Export options
    st.markdown("---")
    st.markdown("### Export Results")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export CSV"):
            try:
                import pandas as pd
                from pathlib import Path

                output_dir = Path("./output/tank")
                output_dir.mkdir(parents=True, exist_ok=True)

                if results.get('ox'):
                    df_ox = pd.DataFrame({
                        'time_s': results['ox']['t'],
                        'pressure_bar': results['ox']['p'] / 1e5,
                        'temperature_K': results['ox']['T'],
                        'mass_kg': results['ox']['m_liquid']
                    })
                    df_ox.to_csv(output_dir / "oxidizer_tank.csv", index=False)

                if results.get('fuel'):
                    df_fuel = pd.DataFrame({
                        'time_s': results['fuel']['t'],
                        'pressure_bar': results['fuel']['p'] / 1e5,
                        'temperature_K': results['fuel']['T'],
                        'mass_kg': results['fuel']['m_liquid']
                    })
                    df_fuel.to_csv(output_dir / "fuel_tank.csv", index=False)

                st.success(f"Exported to {output_dir}")

            except Exception as e:
                st.error(f"Export failed: {e}")

    with col2:
        if st.button("Generate Report"):
            st.info("Report generation would be implemented here.")
