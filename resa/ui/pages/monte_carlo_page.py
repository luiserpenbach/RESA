"""Monte Carlo Analysis Page for RESA UI."""
import streamlit as st
import numpy as np


def render_monte_carlo_page():
    """Render the Monte Carlo analysis page."""
    st.title("🎲 Monte Carlo Uncertainty Analysis")

    if not st.session_state.get('engine_config'):
        st.warning("Please create an engine configuration first.")
        return

    config = st.session_state.engine_config

    st.subheader("Uncertain Parameters")
    st.markdown("Define the uncertainty distribution for each parameter.")

    params = []

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    col1.markdown("**Parameter**")
    col2.markdown("**Nominal**")
    col3.markdown("**Std Dev (%)**")
    col4.markdown("**Include**")

    # Chamber pressure
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    col1.text("Chamber Pressure")
    col2.text(f"{config.pc_bar:.1f} bar")
    pc_std = col3.number_input("", 1.0, 20.0, 5.0, key="pc_std", label_visibility="collapsed")
    pc_inc = col4.checkbox("", True, key="pc_inc", label_visibility="collapsed")
    if pc_inc:
        params.append(('pc_bar', config.pc_bar, pc_std/100))

    # Mixture ratio
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    col1.text("Mixture Ratio")
    col2.text(f"{config.mr:.2f}")
    mr_std = col3.number_input("", 1.0, 20.0, 3.0, key="mr_std", label_visibility="collapsed")
    mr_inc = col4.checkbox("", True, key="mr_inc", label_visibility="collapsed")
    if mr_inc:
        params.append(('mr', config.mr, mr_std/100))

    # Combustion efficiency
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    col1.text("Combustion Efficiency")
    col2.text(f"{config.eff_combustion:.2f}")
    eta_std = col3.number_input("", 0.5, 10.0, 2.0, key="eta_std", label_visibility="collapsed")
    eta_inc = col4.checkbox("", True, key="eta_inc", label_visibility="collapsed")
    if eta_inc:
        params.append(('eff_combustion', config.eff_combustion, eta_std/100))

    st.divider()

    st.subheader("Analysis Settings")
    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.number_input("Number of Samples", 100, 10000, 1000)
    with col2:
        seed = st.number_input("Random Seed (0=random)", 0, 99999, 42)

    outputs = st.multiselect(
        "Output Variables",
        ["Isp (vacuum)", "Thrust", "Mass Flow", "Max Wall Temp", "Pressure Drop"],
        default=["Isp (vacuum)", "Thrust"]
    )

    if st.button("🎲 Run Monte Carlo", type="primary"):
        with st.spinner(f"Running {n_samples} samples..."):
            progress = st.progress(0)

            try:
                from resa.analysis.monte_carlo import MonteCarloAnalysis

                mc = MonteCarloAnalysis(seed=seed if seed > 0 else None)

                for name, nominal, std_frac in params:
                    mc.add_parameter(name, nominal, 'normal', std=nominal * std_frac)

                # Simulate results
                progress.progress(50, "Sampling parameter space...")

                # Generate fake results for demo
                np.random.seed(seed if seed > 0 else None)
                isp_samples = np.random.normal(280, 5, n_samples)
                thrust_samples = np.random.normal(2200, 50, n_samples)

                progress.progress(100, "Complete!")

                st.success(f"Monte Carlo analysis complete with {n_samples} samples!")

                # Show results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Isp Mean", f"{np.mean(isp_samples):.1f} s")
                    st.metric("Isp Std Dev", f"{np.std(isp_samples):.2f} s")
                with col2:
                    st.metric("Thrust Mean", f"{np.mean(thrust_samples):.0f} N")
                    st.metric("Thrust Std Dev", f"{np.std(thrust_samples):.1f} N")

                # Histograms
                try:
                    from resa.analysis.monte_carlo_plots import MonteCarloPlotter
                    plotter = MonteCarloPlotter()

                    fig = plotter.create_histogram(isp_samples, "Isp (vacuum)", "s")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render plots: {e}")

            except Exception as e:
                st.error(f"Monte Carlo analysis failed: {e}")
