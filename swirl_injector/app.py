#!/usr/bin/env python3
"""
Swirl Injector Dimensioning Tool - Streamlit UI

A web-based interface for designing and analyzing swirl coaxial injectors
for liquid rocket engines.

Run with: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import json
import yaml

# Add module to path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    InjectorConfig, PropellantConfig, OperatingConditions,
    GeometryConfig, ColdFlowConfig
)
from results import InjectorResults
from calculators import LCSCCalculator, GCSCCalculator, ColdFlowCalculator
from thermodynamics import (
    DischargeCoefficients, SprayAngleCorrelations,
    FilmThicknessCorrelations, COOLPROP_AVAILABLE
)

# Try to import plotly for interactive charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Swirl Injector Design Tool",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""

    # Header
    st.markdown('<p class="main-header">🚀 Swirl Injector Design Tool</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">LCSC & GCSC Injector Dimensioning for Liquid Rocket Engines</p>',
                unsafe_allow_html=True)

    # Show CoolProp status
    if not COOLPROP_AVAILABLE:
        st.warning(
            "⚠️ CoolProp not available. Using approximate fluid properties. Install CoolProp for accurate calculations.")

    # Sidebar for inputs
    with st.sidebar:
        st.header("⚙️ Design Parameters")

        # Injector type selection
        injector_type = st.selectbox(
            "Injector Type",
            ["LCSC (Liquid-Centered)", "GCSC (Gas-Centered)"],
            help="LCSC: Liquid fuel swirled around central gas oxidizer\nGCSC: Gas oxidizer flows through center of swirling liquid"
        )
        is_lcsc = "LCSC" in injector_type

        st.divider()

        # Propellant properties
        st.subheader("🧪 Propellants")

        col1, col2 = st.columns(2)
        with col1:
            fuel_options = ["Ethanol", "RP-1", "Methanol", "MMH"]
            fuel_selection = st.selectbox("Fuel", fuel_options)
            fuel_temp = st.number_input("Fuel Temperature (K)",
                                        value=300.0, min_value=200.0, max_value=400.0, step=10.0)

        with col2:
            ox_options = ["Nitrous Oxide", "LOX", "N2O4", "H2O2"]
            ox_selection = st.selectbox("Oxidizer", ox_options)
            ox_temp = st.number_input("Oxidizer Temperature (K)",
                                      value=300.0, min_value=100.0, max_value=600.0, step=10.0)

        # Map selections to CoolProp names
        fuel_map = {
            "Ethanol": "REFPROP::Ethanol",
            "RP-1": "REFPROP::Ethanol",  # Approximation
            "Methanol": "REFPROP::Methanol",
            "MMH": "REFPROP::Ethanol",  # Approximation
        }
        ox_map = {
            "Nitrous Oxide": "REFPROP::NitrousOxide",
            "LOX": "REFPROP::Oxygen",
            "N2O4": "REFPROP::NitrousOxide",  # Approximation
            "H2O2": "REFPROP::Water",  # Approximation
        }

        st.divider()

        # Operating conditions
        st.subheader("📊 Operating Conditions")

        inlet_pressure = st.number_input(
            "Inlet Pressure (bar)",
            value=45.0, min_value=10.0, max_value=200.0, step=5.0
        ) * 1e5

        pressure_drop = st.number_input(
            "Pressure Drop (bar)",
            value=20.0, min_value=1.0, max_value=50.0, step=1.0
        ) * 1e5

        col1, col2 = st.columns(2)
        with col1:
            mass_flow_fuel = st.number_input(
                "Fuel Mass Flow (kg/s)",
                value=0.20, min_value=0.01, max_value=10.0, step=0.01, format="%.3f"
            )
        with col2:
            mass_flow_ox = st.number_input(
                "Oxidizer Mass Flow (kg/s)",
                value=0.80, min_value=0.01, max_value=50.0, step=0.01, format="%.3f"
            )

        ox_velocity = st.number_input(
            "Oxidizer Velocity (m/s)",
            value=100.0, min_value=10.0, max_value=500.0, step=10.0,
            help="Target oxidizer exit velocity from injector"
        )

        st.divider()

        # Geometry
        st.subheader("📐 Geometry")

        col1, col2 = st.columns(2)
        with col1:
            num_elements = st.number_input(
                "Number of Elements",
                value=3, min_value=1, max_value=50, step=1
            )
            num_fuel_ports = st.number_input(
                "Fuel Ports per Element",
                value=3, min_value=1, max_value=8, step=1,
                help="Number of tangential fuel inlet ports"
            )

        with col2:
            num_ox_orifices = st.number_input(
                "Ox Orifices per Element",
                value=1, min_value=1, max_value=8, step=1,
                help="Number of oxidizer inlet orifices per element"
            )
            post_thickness = st.number_input(
                "Post Thickness (mm)",
                value=0.5, min_value=0.1, max_value=5.0, step=0.1
            ) * 1e-3

        if is_lcsc:
            spray_half_angle = st.slider(
                "Spray Half Angle (°)",
                min_value=30, max_value=80, value=60, step=5,
                help="Design parameter for LCSC injectors"
            )
        else:
            spray_half_angle = 60  # Not used for GCSC
            min_clearance = st.number_input(
                "Minimum Clearance (mm)",
                value=0.5, min_value=0.1, max_value=2.0, step=0.1,
                help="Manufacturing constraint for GCSC"
            ) * 1e-3

    # Build configuration
    config = InjectorConfig(
        propellants=PropellantConfig(
            fuel=fuel_map.get(fuel_selection, "REFPROP::Ethanol"),
            oxidizer=ox_map.get(ox_selection, "REFPROP::NitrousOxide"),
            fuel_temperature=fuel_temp,
            oxidizer_temperature=ox_temp
        ),
        operating=OperatingConditions(
            inlet_pressure=inlet_pressure,
            pressure_drop=pressure_drop,
            mass_flow_fuel=mass_flow_fuel,
            mass_flow_oxidizer=mass_flow_ox,
            oxidizer_velocity=ox_velocity
        ),
        geometry=GeometryConfig(
            num_elements=num_elements,
            num_fuel_ports=num_fuel_ports,
            num_ox_orifices=num_ox_orifices,
            post_thickness=post_thickness,
            spray_half_angle=float(spray_half_angle),
            minimum_clearance=min_clearance if not is_lcsc else 0.5e-3
        )
    )

    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Results", "Parametric Study", "Cold Flow", "Correlations", "Export"
    ])

    # Calculate results
    try:
        if is_lcsc:
            calc = LCSCCalculator(config)
        else:
            calc = GCSCCalculator(config)

        results = calc.calculate()
        calculation_success = True
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        calculation_success = False
        results = None

    # Tab 1: Results
    with tab1:
        if calculation_success and results:
            display_results(results, config)

    # Tab 2: Parametric Study
    with tab2:
        if calculation_success:
            display_parametric_study(config, is_lcsc)

    # Tab 3: Cold Flow
    with tab3:
        if calculation_success and results:
            display_cold_flow(results, config)

    # Tab 4: Correlations
    with tab4:
        display_correlations()

    # Tab 5: Export
    with tab5:
        if calculation_success and results:
            display_export(results, config)


def display_results(results: InjectorResults, config: InjectorConfig):
    """Display calculation results."""

    st.header("Design Results")

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Momentum Flux Ratio (J)",
            f"{results.performance.momentum_flux_ratio:.2f}",
            help="Ratio of oxidizer to fuel momentum flux"
        )
    with col2:
        st.metric(
            "Weber Number",
            f"{results.performance.weber_number:.0f}",
            help="Ratio of inertial to surface tension forces"
        )
    with col3:
        st.metric(
            "Swirl Number",
            f"{results.performance.swirl_number:.1f}",
            help="Geometric swirl intensity parameter"
        )
    with col4:
        st.metric(
            "Mixture Ratio (O/F)",
            f"{results.mass_flows.mixture_ratio:.2f}",
            help="Oxidizer to fuel mass ratio"
        )

    st.divider()

    # Geometry and performance details
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Geometry")

        geom_data = {
            "Parameter": [
                "Fuel Orifice Ø",
                "Fuel Port Ø",
                "Fuel Port Length",
                "Swirl Chamber Ø",
                "Ox Outlet Ø",
                "Ox Inlet Orifice Ø",
                "Recess Length",
                "Film Thickness",
            ],
            "Value": [
                f"{results.geometry.fuel_orifice_radius * 2000:.3f} mm",
                f"{results.geometry.fuel_port_radius * 2000:.3f} mm",
                f"{results.geometry.fuel_port_length * 1000:.3f} mm",
                f"{results.geometry.swirl_chamber_radius * 2000:.3f} mm",
                f"{results.geometry.ox_outlet_radius * 2000:.3f} mm",
                f"{results.geometry.ox_inlet_orifice_radius * 2000:.3f} mm",
                f"{results.geometry.recess_length * 1000:.3f} mm",
                f"{results.performance.film_thickness * 1000:.3f} mm",
            ]
        }
        st.table(pd.DataFrame(geom_data))

    with col2:
        st.subheader("📊 Performance")

        perf_data = {
            "Parameter": [
                "Spray Half Angle",
                "Discharge Coefficient",
                "Velocity Ratio",
                "Reynolds Number (Port)",
                "Fuel per Element",
                "Oxidizer per Element",
            ],
            "Value": [
                f"{results.performance.spray_half_angle:.1f}°",
                f"{results.performance.discharge_coefficient:.4f}",
                f"{results.performance.velocity_ratio:.2f}",
                f"{results.performance.reynolds_port:.0f}" if results.performance.reynolds_port else "N/A",
                f"{results.mass_flows.fuel_per_element:.4f} kg/s",
                f"{results.mass_flows.oxidizer_per_element:.4f} kg/s",
            ]
        }
        st.table(pd.DataFrame(perf_data))

    # Injector schematic
    st.subheader("Injector Schematic")

    if PLOTLY_AVAILABLE:
        fig = create_injector_schematic(results)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Install Plotly for interactive schematics: `pip install plotly`")

        # Text-based representation
        st.code(f"""
        Injector Element Cross-Section (not to scale)

        ┌─────────────────────┬────┬─────────┐
        │                     │    │         │
        │   Swirl Chamber     │ Or │  Ox     │
        │   Ø{results.geometry.swirl_chamber_radius * 2000:.1f} mm         │ if │  Port   │
        │                     │ ice│         │
        │  ← Tangential       │    │         │
        │    Port             │    │         │
        │    Ø{results.geometry.port_radius * 2000:.2f} mm       │    │         │
        └─────────────────────┴────┴─────────┘
        """)


def display_parametric_study(config: InjectorConfig, is_lcsc: bool):
    """Display parametric study interface."""

    st.header("Parametric Study")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Study Parameters")

        param_options = {
            "Pressure Drop": ("operating", "pressure_drop", 5e5, 35e5, 1e5, "bar", 1e5),
            "Fuel Mass Flow": ("operating", "mass_flow_fuel", 0.05, 0.5, 0.05, "kg/s", 1),
            "Oxidizer Velocity": ("operating", "oxidizer_velocity", 50, 200, 10, "m/s", 1),
            "Number of Elements": ("geometry", "num_elements", 1, 10, 1, "-", 1),
        }

        if is_lcsc:
            param_options["Spray Half Angle"] = ("geometry", "spray_half_angle", 35, 75, 5, "°", 1)

        selected_param = st.selectbox("Parameter to Vary", list(param_options.keys()))

        param_info = param_options[selected_param]
        section, attr, min_val, max_val, step, unit, scale = param_info

        param_range = st.slider(
            f"Range ({unit})",
            min_value=float(min_val / scale),
            max_value=float(max_val / scale),
            value=(float(min_val / scale), float(max_val / scale)),
            step=float(step / scale)
        )

        num_points = st.slider("Number of Points", 5, 20, 10)

    with col2:
        st.subheader("Results")

        # Run parametric study
        values = np.linspace(param_range[0] * scale, param_range[1] * scale, num_points)

        results_data = {
            "Parameter": [],
            "J": [],
            "We": [],
            "Spray Angle (°)": [],
            "Orifice Ø (mm)": [],
        }

        Calculator = LCSCCalculator if is_lcsc else GCSCCalculator

        for val in values:
            try:
                config_dict = config.to_dict()
                config_dict[section][attr] = float(val)
                mod_config = InjectorConfig.from_dict(config_dict)

                calc = Calculator(mod_config)
                result = calc.calculate()

                results_data["Parameter"].append(val / scale)
                results_data["J"].append(result.performance.momentum_flux_ratio)
                results_data["We"].append(result.performance.weber_number)
                results_data["Spray Angle (°)"].append(result.performance.spray_half_angle)
                results_data["Orifice Ø (mm)"].append(result.geometry.orifice_radius * 2000)
            except:
                pass

        df = pd.DataFrame(results_data)

        if PLOTLY_AVAILABLE and len(df) > 0:
            fig = make_subplots(rows=2, cols=2, subplot_titles=["Momentum Flux Ratio", "Weber Number", "Spray Angle",
                                                                "Orifice Diameter"])

            fig.add_trace(go.Scatter(x=df["Parameter"], y=df["J"], mode='lines+markers', name="J"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Parameter"], y=df["We"], mode='lines+markers', name="We"), row=1, col=2)
            fig.add_trace(go.Scatter(x=df["Parameter"], y=df["Spray Angle (°)"], mode='lines+markers', name="α"), row=2,
                          col=1)
            fig.add_trace(go.Scatter(x=df["Parameter"], y=df["Orifice Ø (mm)"], mode='lines+markers', name="Ø"), row=2,
                          col=2)

            fig.update_layout(height=500, showlegend=False)
            fig.update_xaxes(title_text=f"{selected_param} ({unit})")

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(df)


def display_cold_flow(results: InjectorResults, config: InjectorConfig):
    """Display cold flow equivalent calculations."""

    st.header("Cold Flow Test Equivalent")

    st.info("Calculate water/nitrogen test conditions that match hot-fire momentum flux ratio.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cold Flow Conditions")

        cf_inlet_pressure = st.number_input(
            "CF Inlet Pressure (bar)",
            value=21.0, min_value=5.0, max_value=100.0, step=1.0
        ) * 1e5

        cf_pressure_drop = st.number_input(
            "CF Pressure Drop (bar)",
            value=20.0, min_value=1.0, max_value=100.0, step=1.0
        ) * 1e5

        cf_temp = st.number_input(
            "Ambient Temperature (K)",
            value=293.15, min_value=273.0, max_value=323.0, step=5.0
        )

    # Calculate cold flow equivalent
    try:
        cold_config = ColdFlowConfig(
            inlet_pressure=cf_inlet_pressure,
            pressure_drop=cf_pressure_drop,
            ambient_temperature=cf_temp
        )

        cold_calc = ColdFlowCalculator(
            hot_fire_results=results,
            cold_flow_config=cold_config,
            geometry_config=config.geometry
        )

        is_gcsc = results.injector_type == "GCSC"
        cold_results = cold_calc.calculate(is_gcsc=is_gcsc)

        with col2:
            st.subheader("Cold Flow Results")

            cf_data = {
                "Parameter": [
                    "Water Mass Flow",
                    "Nitrogen Mass Flow",
                    "Nitrogen Velocity",
                    "Momentum Flux Ratio",
                    "Weber Number",
                    "Velocity Ratio",
                ],
                "Value": [
                    f"{cold_results.liquid_mass_flow:.4f} kg/s",
                    f"{cold_results.gas_mass_flow:.4f} kg/s",
                    f"{cold_results.gas_velocity:.1f} m/s",
                    f"{cold_results.performance.momentum_flux_ratio:.2f}",
                    f"{cold_results.performance.weber_number:.0f}",
                    f"{cold_results.performance.velocity_ratio:.2f}",
                ]
            }
            st.table(pd.DataFrame(cf_data))

            # Comparison
            st.subheader("Hot vs Cold Comparison")

            comparison_data = {
                "Metric": ["J", "We", "Velocity Ratio"],
                "Hot Fire": [
                    f"{results.performance.momentum_flux_ratio:.2f}",
                    f"{results.performance.weber_number:.0f}",
                    f"{results.performance.velocity_ratio:.2f}",
                ],
                "Cold Flow": [
                    f"{cold_results.performance.momentum_flux_ratio:.2f}",
                    f"{cold_results.performance.weber_number:.0f}",
                    f"{cold_results.performance.velocity_ratio:.2f}",
                ],
            }
            st.table(pd.DataFrame(comparison_data))

    except Exception as e:
        st.error(f"Cold flow calculation error: {str(e)}")


def display_correlations():
    """Display correlation comparison charts."""

    st.header("Correlation Comparison")

    st.markdown("""
    Compare different empirical correlations for discharge coefficient and spray angle
    across a range of swirl numbers.
    """)

    col1, col2 = st.columns(2)

    swirl_range = np.linspace(2, 50, 100)

    # Calculate correlations
    cd_data = {"Swirl Number": swirl_range}
    alpha_data = {"Swirl Number": swirl_range}

    # Back-calculate geometry for each swirl number
    r_sc = 1.0
    n_p = 3

    cd_abramovic = []
    cd_fu = []
    cd_anand = []
    alpha_anand = []

    for SN in swirl_range:
        # Solve for r_p from swirl number
        a = n_p * SN
        b = r_sc
        c = -r_sc ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            r_p = (-b + np.sqrt(discriminant)) / (2 * a)

            cd_abramovic.append(DischargeCoefficients.abramovic(r_sc, r_p, n_p))
            cd_fu.append(DischargeCoefficients.fu(r_sc, r_p, n_p))
            cd_anand.append(DischargeCoefficients.anand(r_sc, r_p, n_p))
            alpha_anand.append(np.rad2deg(SprayAngleCorrelations.anand(r_sc, r_p, n_p, 1.5)))
        else:
            cd_abramovic.append(np.nan)
            cd_fu.append(np.nan)
            cd_anand.append(np.nan)
            alpha_anand.append(np.nan)

    with col1:
        st.subheader("Discharge Coefficient")

        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=swirl_range, y=cd_abramovic, name="Abramovic", mode='lines'))
            fig.add_trace(go.Scatter(x=swirl_range, y=cd_fu, name="Fu et al.", mode='lines'))
            fig.add_trace(go.Scatter(x=swirl_range, y=cd_anand, name="Anand et al.", mode='lines'))
            fig.update_layout(
                xaxis_title="Swirl Number",
                yaxis_title="Discharge Coefficient",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            df = pd.DataFrame({
                "Swirl Number": swirl_range,
                "Abramovic": cd_abramovic,
                "Fu": cd_fu,
                "Anand": cd_anand,
            })
            st.line_chart(df.set_index("Swirl Number"))

    with col2:
        st.subheader("Spray Half Angle")

        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=swirl_range, y=alpha_anand, name="Anand (RR=1.5)", mode='lines'))
            fig.update_layout(
                xaxis_title="Swirl Number",
                yaxis_title="Spray Half Angle (°)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            df = pd.DataFrame({
                "Swirl Number": swirl_range,
                "Anand": alpha_anand,
            })
            st.line_chart(df.set_index("Swirl Number"))

    # Correlation references
    with st.expander("Correlation References"):
        st.markdown("""
        **Discharge Coefficient Correlations:**
        - **Abramovic**: $C_D = 0.432 \\cdot A^{-0.64}$
        - **Fu et al.**: $C_D = 0.4354 \\cdot A^{-0.877}$
        - **Anand et al.**: $C_D = 1.28 \\cdot A^{-1.28}$

        **Spray Angle Correlations:**
        - **Anand et al.**: $\\alpha = \\arctan(0.01 \\cdot A^{1.64} \\cdot RR^{-0.242})$
        - **Lefebvre**: $\\alpha = \\arcsin\\left(\\frac{X\\sqrt{8}}{1 + \\sqrt{X}\\sqrt{1+X}}\\right)$

        Where $A$ is the swirl number and $X$ is the open area ratio.
        """)


def display_export(results: InjectorResults, config: InjectorConfig):
    """Display export options."""

    st.header("Export Design")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Configuration")

        # YAML export
        yaml_str = yaml.dump(config.to_dict(), default_flow_style=False)
        st.download_button(
            "Download Config (YAML)",
            yaml_str,
            file_name="injector_config.yaml",
            mime="text/yaml"
        )

        # JSON export
        json_str = json.dumps(config.to_dict(), indent=2)
        st.download_button(
            "Download Config (JSON)",
            json_str,
            file_name="injector_config.json",
            mime="application/json"
        )

        st.text_area("Configuration Preview", yaml_str, height=300)

    with col2:
        st.subheader("📊 Results")

        # Results JSON
        results_dict = {
            "injector_type": results.injector_type,
            "geometry": {
                "fuel_orifice_radius_mm": results.geometry.fuel_orifice_radius * 1000,
                "fuel_port_radius_mm": results.geometry.fuel_port_radius * 1000,
                "fuel_port_length_mm": results.geometry.fuel_port_length * 1000,
                "swirl_chamber_radius_mm": results.geometry.swirl_chamber_radius * 1000,
                "ox_outlet_radius_mm": results.geometry.ox_outlet_radius * 1000,
                "ox_inlet_orifice_radius_mm": results.geometry.ox_inlet_orifice_radius * 1000,
                "recess_length_mm": results.geometry.recess_length * 1000,
            },
            "performance": {
                "spray_half_angle_deg": results.performance.spray_half_angle,
                "swirl_number": results.performance.swirl_number,
                "momentum_flux_ratio": results.performance.momentum_flux_ratio,
                "weber_number": results.performance.weber_number,
                "discharge_coefficient": results.performance.discharge_coefficient,
                "velocity_ratio": results.performance.velocity_ratio,
                "film_thickness_mm": results.performance.film_thickness * 1000 if results.performance.film_thickness else None,
            },
            "mass_flows": {
                "fuel_per_element_kg_s": results.mass_flows.fuel_per_element,
                "oxidizer_per_element_kg_s": results.mass_flows.oxidizer_per_element,
                "mixture_ratio": results.mass_flows.mixture_ratio,
            }
        }

        results_json = json.dumps(results_dict, indent=2)
        st.download_button(
            "Download Results (JSON)",
            results_json,
            file_name="injector_results.json",
            mime="application/json"
        )

        st.text_area("Results Preview", results_json, height=300)

    # Summary report
    st.subheader("📋 Summary Report")

    report = f"""
# Swirl Injector Design Report
## {results.injector_type} Configuration

### Operating Conditions
- Inlet Pressure: {config.operating.inlet_pressure / 1e5:.1f} bar
- Pressure Drop: {config.operating.pressure_drop / 1e5:.1f} bar
- Fuel Mass Flow: {config.operating.mass_flow_fuel:.3f} kg/s
- Oxidizer Mass Flow: {config.operating.mass_flow_oxidizer:.3f} kg/s
- Mixture Ratio: {results.mass_flows.mixture_ratio:.2f}

### Geometry
- Number of Elements: {config.geometry.num_elements}
- Fuel Ports per Element: {config.geometry.num_fuel_ports}
- Ox Orifices per Element: {config.geometry.num_ox_orifices}

#### Fuel Side
- Fuel Orifice Diameter: {results.geometry.fuel_orifice_radius * 2000:.3f} mm
- Fuel Port Diameter: {results.geometry.fuel_port_radius * 2000:.3f} mm
- Fuel Port Length: {results.geometry.fuel_port_length * 1000:.3f} mm
- Swirl Chamber Diameter: {results.geometry.swirl_chamber_radius * 2000:.3f} mm

#### Oxidizer Side
- Ox Outlet Diameter: {results.geometry.ox_outlet_radius * 2000:.3f} mm
- Ox Inlet Orifice Diameter: {results.geometry.ox_inlet_orifice_radius * 2000:.3f} mm

#### Mixing
- Recess Length: {results.geometry.recess_length * 1000:.3f} mm
- Film Thickness: {results.performance.film_thickness * 1000:.3f} mm

### Performance
- Spray Half Angle: {results.performance.spray_half_angle:.1f} deg
- Swirl Number: {results.performance.swirl_number:.2f}
- Momentum Flux Ratio (J): {results.performance.momentum_flux_ratio:.3f}
- Weber Number: {results.performance.weber_number:.0f}
- Velocity Ratio: {results.performance.velocity_ratio:.2f}
- Discharge Coefficient: {results.performance.discharge_coefficient:.4f}
"""

    st.download_button(
        "Download Report (Markdown)",
        report,
        file_name="injector_report.md",
        mime="text/markdown"
    )

    st.markdown(report)


def create_injector_schematic(results: InjectorResults) -> "go.Figure":
    """Create an injector cross-section schematic."""

    geom = results.geometry

    # Convert to mm
    r_fuel_o = geom.fuel_orifice_radius * 1000
    r_sc = geom.swirl_chamber_radius * 1000
    r_ox_out = geom.ox_outlet_radius * 1000
    r_fuel_p = geom.fuel_port_radius * 1000

    l_fuel_o = r_fuel_o  # Estimate orifice length
    l_sc = 2 * r_sc  # Estimate swirl chamber length

    fig = go.Figure()

    # Swirl chamber (outer boundary)
    fig.add_shape(type="rect",
                  x0=0, y0=-r_sc, x1=l_sc, y1=r_sc,
                  line=dict(color="blue", width=2),
                  fillcolor="rgba(173, 216, 230, 0.3)"
                  )

    # Fuel orifice
    fig.add_shape(type="rect",
                  x0=l_sc, y0=-r_fuel_o, x1=l_sc + l_fuel_o, y1=r_fuel_o,
                  line=dict(color="blue", width=2),
                  fillcolor="rgba(173, 216, 230, 0.5)"
                  )

    # Oxidizer outlet
    fig.add_shape(type="rect",
                  x0=l_sc + l_fuel_o, y0=-r_ox_out, x1=l_sc + l_fuel_o + 5, y1=r_ox_out,
                  line=dict(color="red", width=2),
                  fillcolor="rgba(255, 200, 200, 0.3)"
                  )

    # Tangential fuel port indicator
    fig.add_shape(type="circle",
                  x0=-r_fuel_p, y0=r_sc - 2 * r_fuel_p, x1=r_fuel_p, y1=r_sc,
                  line=dict(color="green", width=2),
                  fillcolor="rgba(144, 238, 144, 0.5)"
                  )

    # Annotations
    fig.add_annotation(x=l_sc / 2, y=r_sc + 2, text=f"Swirl Chamber Ø{2 * r_sc:.1f}mm",
                       showarrow=False, font=dict(size=10))
    fig.add_annotation(x=l_sc + l_fuel_o / 2, y=-r_fuel_o - 2, text=f"Fuel Orifice Ø{2 * r_fuel_o:.2f}mm",
                       showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0, y=r_sc - r_fuel_p, text=f"Fuel Port Ø{2 * r_fuel_p:.2f}mm",
                       showarrow=True, ax=-30, ay=0, font=dict(size=10))

    fig.update_layout(
        title=f"{results.injector_type} Injector Element Cross-Section",
        xaxis_title="Axial Position (mm)",
        yaxis_title="Radial Position (mm)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=False,
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


if __name__ == "__main__":
    main()