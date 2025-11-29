import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import your library
# (Ensure 'src' is in the python path or run streamlit from the root folder)
from rocket_engine import LiquidEngine, EngineConfig
from rocket_engine import plot_n2o_t_rho_diagram, plot_n2o_p_t_diagram
from rocket_engine import plot_channel_cross_section_radial
from rocket_engine import plot_coolant_channels_3d

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RESA: Rocket Engine Sizing & Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("HOPPER RESA Design Suite")
st.markdown("---")

# --- SIDEBAR: CONFIGURATION ---
st.sidebar.header("1. Engine Inputs")

# General Settings
with st.sidebar.expander("General Configuration", expanded=True):
    engine_name = st.text_input("Engine Name", "Phoenix-1B")
    col1, col2 = st.columns(2)
    thrust = col1.number_input("Thrust [N]", 100.0, 100000.0, 2200.0)
    pc = col2.number_input("Chamber Pressure [bar]", 1.0, 200.0, 25.0)

    col3, col4 = st.columns(2)
    mr = col3.number_input("Mixture Ratio (O/F)", 0.5, 10.0, 4.0)
    eff = col4.number_input("Combustion Eff.", 0.8, 1.0, 0.95)

    fuel = st.selectbox("Fuel", ["Ethanol90", "Ethanol", "RP1", "LH2", "CH4"])
    oxidizer = st.selectbox("Oxidizer", ["N2O", "LOX", "NitrousOxide"])

# Nozzle Settings
with st.sidebar.expander("Chamber & Nozzle"):
    col1, col2 = st.columns(2)
    eps = col1.number_input("Expansion Ratio (0=Opt)", 0.0, 200.0, 0.0)
    p_exit = col2.number_input("Design Exit P [bar]", 0.0, 5.0, 1.013)

    L_star = st.slider("L* (Char. Length) [mm]", 500.0, 2000.0, 1100.0)
    contr_ratio = st.slider("Contraction Ratio", 2.0, 20.0, 10.0)
    bell_frac = st.slider("Bell Fraction (%)", 60, 100, 80) / 100.0

# Cooling Settings
with st.sidebar.expander("Cooling System"):
    coolant = st.text_input("Coolant String", "REFPROP::NitrousOxide")
    mode = st.radio("Flow Mode", ["counter-flow", "co-flow"])

    st.subheader("Geometry (at Throat)")
    c1, c2 = st.columns(2)
    w_ch = c1.number_input("Channel Width [mm]", 0.1, 10.0, 1.0) / 1000.0
    h_ch = c2.number_input("Channel Height [mm]", 0.1, 10.0, 0.75) / 1000.0
    rib = st.number_input("Rib Width [mm]", 0.1, 10.0, 0.6) / 1000.0

    t_wall = st.number_input("Wall Thickness [mm]", 0.1, 5.0, 0.5) / 1000.0

    st.subheader("Inlet State")
    p_in = st.number_input("Coolant P_in [bar]", 10.0, 200.0, 60.0)
    t_in = st.number_input("Coolant T_in [K]", 50.0, 500.0, 285.0)

# --- INITIALIZE ENGINE OBJECT ---
config = EngineConfig(
    engine_name=engine_name,
    fuel=fuel,
    oxidizer=oxidizer,
    thrust_n=thrust,
    pc_bar=pc,
    mr=mr,
    expansion_ratio=eps,
    p_exit_bar=p_exit,
    L_star=L_star,
    contraction_ratio=contr_ratio,
    eff_combustion=eff,
    bell_fraction=bell_frac,
    coolant_name=coolant,
    cooling_mode=mode,
    channel_width_throat=w_ch,
    channel_height=h_ch,
    rib_width_throat=rib,
    wall_thickness=t_wall,
    coolant_p_in_bar=p_in,
    coolant_t_in_k=t_in
)

engine = LiquidEngine(config)
# If we have a saved design in history, load its geometry
# into the current engine instance so 'analyze' works.
if 'design_result' in st.session_state:
    saved_res = st.session_state['design_result']

    # Restore the specific geometry objects required by engine.analyze()
    engine.geo = saved_res.geometry
    engine.channel_geo = saved_res.channel_geometry

    # Optional: Inform user that geometry is locked
    st.sidebar.success(f"Locked to Geometry: {saved_res.geometry.x_full.shape[0]} Nodes")

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["🏗️ Design Point", "📉 Off-Design Analysis", "⏱️ Transient"])

# ... [Previous imports and setup code] ...

# =========================================
# TAB 1: DESIGN POINT
# =========================================
with tab1:
    if st.button("Run Design Sizing", type="primary"):
        with st.spinner("Calculating Combustion, Geometry, and Cooling..."):
            res = engine.design(plot=False)

            # 1. Top Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Vacuum Isp", f"{res.isp_vac:.1f} s")
            m2.metric("Thrust (Vac)", f"{res.thrust_vac:.0f} N")
            m3.metric("Mass Flow", f"{res.massflow_total:.3f} kg/s")
            m4.metric("Expansion Ratio", f"{res.expansion_ratio:.1f}")

            # 2. Main Plots
            st.subheader("Performance Analysis")
            col_plot1, col_plot2 = st.columns([2, 1])

            with col_plot1:
                # -- Dashboard Plot --
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

                xs = res.geometry.x_full * 1000
                ys = res.geometry.y_full

                # Geometry & Wall Temp
                ax1.plot(xs, ys, 'k-', lw=2, label="Contour")
                ax1_r = ax1.twinx()
                ax1_r.plot(xs, res.cooling_data['T_wall_hot'], 'r-', label="Hot Wall")
                ax1_r.plot(xs, res.cooling_data['T_coolant'], 'b--', label="Coolant")
                ax1_r.set_ylabel("Temp [K]")
                ax1.set_ylabel("Radius [mm]")
                ax1.legend(loc='upper left')
                ax1_r.legend(loc='upper right')
                ax1.grid(True, alpha=0.3)

                # Mach & Flux
                ax2.plot(xs, res.mach_numbers, 'g-', label="Mach")
                ax2_r = ax2.twinx()
                ax2_r.plot(xs, res.cooling_data['q_flux'] / 1e6, 'm-', label="Heat Flux")
                ax2_r.set_ylabel("Flux [MW/m2]")
                ax2.legend(loc='upper left')
                ax2_r.legend(loc='upper right')
                ax2.grid(True, alpha=0.3)

                # Pressure & Density
                ax3.plot(xs, res.cooling_data['P_coolant'] / 1e5, 'c-', label="Pressure")
                ax3_r = ax3.twinx()
                ax3_r.plot(xs, res.cooling_data['density'], 'k:', label="Density")
                ax3_r.set_ylabel("Density [kg/m3]")
                ax3.set_ylabel("Pressure [bar]")
                ax3.set_xlabel("Axial Position [mm]")
                ax3_r.spines["right"].set_position(("axes", 1.1))  # Offset if needed
                ax3.grid(True, alpha=0.3)

                st.pyplot(fig)

            with col_plot2:
                st.write("**Throat Cross Section**")
                # Find throat index
                idx_throat = np.argmin(res.geometry.y_full)
                fig_throat = plot_channel_cross_section_radial(
                    res.channel_geometry,
                    station_idx=idx_throat,
                    closeout_thickness=0.001,
                    sector_angle=90,
                    show=False
                )
                st.pyplot(fig_throat)

                st.write("**Coolant State**")
                try:
                    fig_pt = plot_n2o_p_t_diagram(res, show=False)
                    st.pyplot(fig_pt)
                except Exception as e:
                    st.warning(f"Could not plot Phase Diagram: {e}")

            # 3. DETAILED PARAMETER TABLE
            st.subheader("📋 Specification Sheet")
            with st.expander("View All Parameters", expanded=True):
                # Organize data into a clean structure
                spec_data = [
                    # Category, Parameter, Value, Unit
                    ("Operating Point", "Chamber Pressure", res.pc_bar, "bar"),
                    ("Operating Point", "Mixture Ratio", res.mr, "-"),
                    ("Operating Point", "Total Mass Flow", res.massflow_total, "kg/s"),
                    ("Performance", "Thrust (Vacuum)", res.thrust_vac, "N"),
                    ("Performance", "Thrust (Sea Level)", res.thrust_sea, "N"),
                    ("Performance", "Isp (Vacuum)", res.isp_vac, "s"),
                    ("Performance", "Isp (Sea Level)", res.isp_sea, "s"),
                    #("Performance", "C* (Characteristic Vel)", engine.comb_data.cstar, "m/s"),
                    # Access from engine state
                    ("Geometry", "Throat Diameter", res.dt_mm, "mm"),
                    ("Geometry", "Exit Diameter", res.de_mm, "mm"),
                    ("Geometry", "Expansion Ratio", res.expansion_ratio, "-"),
                    ("Geometry", "Chamber Length", res.length_mm, "mm"),
                    ("Cooling", "Max Wall Temp", np.max(res.cooling_data['T_wall_hot']), "K"),
                    ("Cooling", "Max Coolant Temp", np.max(res.cooling_data['T_coolant']), "K"),
                    ("Cooling", "Max Heat Flux", np.max(res.cooling_data['q_flux']) / 1e6, "MW/m²"),
                    ("Cooling", "Pressure Drop",
                     (np.max(res.cooling_data['P_coolant']) - np.min(res.cooling_data['P_coolant'])) / 1e5, "bar"),
                    ("Cooling", "Min Density", np.min(res.cooling_data['density']), "kg/m³"),
                    ("Cooling", "Coolant Velocity (Exit)", res.cooling_data['velocity'][-1], "m/s")
                ]

                # Create DataFrame
                df_spec = pd.DataFrame(spec_data, columns=["Category", "Parameter", "Value", "Unit"])

                # Formatting: Round values nicely
                # Using styling to display
                st.dataframe(
                    df_spec.style.format({"Value": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True
                )

                # Download Button for this table
                csv = df_spec.to_csv(index=False)
                st.download_button("Download Spec Sheet (CSV)", csv, "engine_spec.csv", "text/csv")

            # Store result in session state for other tabs
            st.session_state['design_result'] = res

# =========================================
# TAB 2: OFF-DESIGN
# =========================================
with tab2:
    st.header("Throttle & Sensitivity Analysis")

    if 'design_result' not in st.session_state:
        st.warning("Please run the Design point in Tab 1 first.")
    else:
        c1, c2 = st.columns(2)
        new_pc = c1.slider("Throttled Chamber Pressure [bar]", 5.0, 50.0, 15.0)
        new_mr = c2.slider("Off-Design O/F Ratio", 1.0, 8.0, 4.0)

        if st.button("Run Off-Design"):
            res_off = engine.analyze(pc_bar=new_pc, mr=new_mr, plot=False)

            st.success(f"Simulation Complete. New Thrust: {res_off.thrust_sea:.1f} N")

            # Comparison Plot
            fig, ax = plt.subplots()
            ax.plot(res_off.geometry.x_full * 1000, res_off.cooling_data['T_wall_hot'], 'r-', label="Off-Design Wall T")
            design_res = st.session_state['design_result']
            ax.plot(design_res.geometry.x_full * 1000, design_res.cooling_data['T_wall_hot'], 'k--',
                    label="Design Wall T")
            ax.set_ylabel("Wall Temp [K]")
            ax.legend()
            st.pyplot(fig)

# =========================================
# TAB 3: TRANSIENT (Future)
# =========================================
with tab3:
    st.header("Startup Transient Simulation")
    st.info("This feature requires the `transients` module to be fully linked.")

    if st.button("Run Startup (0-0.5s)"):
        # Placeholder for integration
        # sol = engine.analyze_transient(0.5)
        st.write("Simulating valve opening...")
        # st.line_chart(...)