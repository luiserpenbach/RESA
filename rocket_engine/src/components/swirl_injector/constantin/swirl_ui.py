import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, BaseLoader


# --- 1. CORE PHYSICS FUNCTIONS (The logic we discussed previously) ---
def calculate_injector(mdot, delta_p, rho=1000):
    # Simplified placeholder logic for demo
    ideal_v = np.sqrt(2 * delta_p / rho)
    # Assume a generic Cd for sizing
    cd_assumed = 0.3
    area_out = mdot / (cd_assumed * np.sqrt(2 * rho * delta_p))
    r_out = np.sqrt(area_out / np.pi)
    return {"r_out": r_out, "d_out_mm": 2 * r_out * 1000, "v_ideal": ideal_v}


def off_design_curve(geometry, pressure_range):
    # Placeholder: C_d changes slightly with pressure in real life
    # but strictly theoretically it is constant for inviscid flow.
    # Let's simulate a viscous loss that drops Cd at low pressures.
    rho = 1000
    flows = []

    for p in pressure_range:
        # Fake viscous curve
        viscous_factor = 1.0 - (1e4 / p)
        cd_effective = 0.3 * viscous_factor

        mdot = cd_effective * (np.pi * geometry['r_out'] ** 2) * np.sqrt(2 * rho * p)
        flows.append(mdot)

    return pd.DataFrame({"Delta_P_Pa": pressure_range, "Mass_Flow": flows})


# --- 2. GUI LAYOUT ---
st.set_page_config(page_title="Swirl Injector Design Tool", layout="wide")

st.title("🚀 Simplex Swirl Injector Designer")

# Create Tabs for the workflow
tab1, tab2, tab3, tab4 = st.tabs(["📐 Design Sizing", "📈 Off-Design Analysis", "🧪 Test Validation", "📄 Report Gen"])

# --- TAB 1: SIZING ---
with tab1:
    st.header("Design Point Inputs")
    col1, col2 = st.columns(2)
    with col1:
        target_mdot = st.number_input("Target Mass Flow (kg/s)", value=0.05, step=0.001, format="%.3f")
        rho = st.number_input("Fluid Density (kg/m3)", value=1000.0)
    with col2:
        delta_p = st.number_input("Target Delta P (Pa)", value=500000.0, step=10000.0)
        target_angle = st.slider("Target Spray Angle", 60, 120, 80)

    if st.button("Run Sizing", type="primary"):
        # Run Calculation
        result = calculate_injector(target_mdot, delta_p, rho)

        # Store in Session State so other tabs can see it
        st.session_state['geometry'] = result
        st.session_state['design_inputs'] = {'mdot': target_mdot, 'dp': delta_p}

        st.success("Sizing Complete!")

        # Display Results
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Required Orifice Diameter", f"{result['d_out_mm']:.3f} mm")
        res_col2.metric("Ideal Velocity", f"{result['v_ideal']:.1f} m/s")

# --- TAB 2: OFF-DESIGN ---
with tab2:
    st.header("Off-Design Performance")

    if 'geometry' not in st.session_state:
        st.warning("Please run Sizing in Tab 1 first.")
    else:
        # Create a pressure range from 1 bar to 20 bar
        p_range = np.linspace(100000, 2000000, 50)
        df_perf = off_design_curve(st.session_state['geometry'], p_range)

        # Store for report
        st.session_state['perf_data'] = df_perf

        # Plotly Interactive Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_perf['Delta_P_Pa'] / 1e5, y=df_perf['Mass_Flow'] * 1000,
                                 mode='lines', name='Theoretical Model'))

        # Add the design point marker
        dp_in = st.session_state['design_inputs']
        fig.add_trace(go.Scatter(x=[dp_in['dp'] / 1e5], y=[dp_in['mdot'] * 1000],
                                 mode='markers', name='Design Point', marker=dict(size=12, color='red')))

        fig.update_layout(title="Mass Flow vs Pressure Drop",
                          xaxis_title="Pressure Drop (Bar)",
                          yaxis_title="Mass Flow (g/s)")

        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: TEST VALIDATION ---
with tab3:
    st.header("Import Cold Flow Data")
    uploaded_file = st.file_uploader("Upload CSV (Columns: pressure_bar, mdot_g_s)", type="csv")

    if uploaded_file is not None and 'perf_data' in st.session_state:
        df_test = pd.read_csv(uploaded_file)

        # Re-plot Theoretical vs Actual
        fig_val = go.Figure()

        # Theoretical
        df_perf = st.session_state['perf_data']
        fig_val.add_trace(go.Scatter(x=df_perf['Delta_P_Pa'] / 1e5, y=df_perf['Mass_Flow'] * 1000,
                                     mode='lines', name='Theoretical Model', line=dict(dash='dash')))

        # Actual Test Data
        fig_val.add_trace(go.Scatter(x=df_test['pressure_bar'], y=df_test['mdot_g_s'],
                                     mode='markers', name='Test Data (Cold Flow)'))

        st.plotly_chart(fig_val, use_container_width=True)

        # Calculate Deviation
        # (Simple interpolation could be added here to calc % Error)

# --- TAB 4: REPORT GENERATION ---
with tab4:
    st.header("Generate Design Report")

    report_html_template = """
    <h1>Injector Design Report</h1>
    <p><b>Date:</b> {{ date }}</p>
    <h2>1. Design Inputs</h2>
    <ul>
        <li>Target Flow: {{ mdot }} kg/s</li>
        <li>Target Pressure: {{ dp }} Pa</li>
    </ul>
    <h2>2. Geometry Output</h2>
    <ul>
        <li>Orifice Diameter: {{ d_out }} mm</li>
    </ul>
    <h2>3. Performance Summary</h2>
    <p>Analysis shows stable operation across {{ range_min }} to {{ range_max }} bar.</p>
    """

    if st.button("Generate PDF Report"):
        if 'geometry' in st.session_state:
            # Render HTML
            template = Environment(loader=BaseLoader()).from_string(report_html_template)
            html_out = template.render(
                date="2023-10-27",
                mdot=st.session_state['design_inputs']['mdot'],
                dp=st.session_state['design_inputs']['dp'],
                d_out=st.session_state['geometry']['d_out_mm'],
                range_min="1", range_max="20"
            )

            # Show HTML preview
            st.markdown("### Preview")
            st.components.v1.html(html_out, height=400, scrolling=True)

            st.info("To save as PDF, standard practice is to use the 'weasyprint' library on this HTML string.")
        else:
            st.error("Run sizing first.")