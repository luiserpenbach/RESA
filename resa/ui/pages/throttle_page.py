"""Throttle Mapping Page for RESA UI."""
import streamlit as st
import numpy as np


def render_throttle_page():
    """Render the throttle mapping page."""
    st.title("⚡ Throttle Map")

    if not st.session_state.get('design_result'):
        st.warning("Please run an engine design first.")
        return

    st.subheader("Throttle Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        min_throttle = st.slider("Min Throttle [%]", 20, 80, 50)
    with col2:
        max_throttle = st.slider("Max Throttle [%]", 80, 100, 100)
    with col3:
        n_points = st.number_input("Number of Points", 5, 21, 11)

    throttle_mode = st.selectbox(
        "Throttling Mode",
        ["Oxidizer Only (Deep Throttle)", "Both Proportional", "Fuel Only"]
    )

    if st.button("🚀 Generate Throttle Map", type="primary"):
        with st.spinner("Generating throttle curve..."):
            # Generate sample data
            throttle_pcts = np.linspace(min_throttle, max_throttle, n_points)
            pc_values = throttle_pcts / 100 * st.session_state.design_result.pc_bar

            try:
                from resa.visualization.performance_plots import ThrottleCurvePlotter

                # Sample data structure
                throttle_data = []
                for pct, pc in zip(throttle_pcts, pc_values):
                    throttle_data.append({
                        'pct': pct,
                        'pc': pc,
                        'mr': st.session_state.design_result.mr * (pct/100 if "Oxidizer" in throttle_mode else 1),
                        'thrust': st.session_state.design_result.thrust_vac * (pct/100)**1.5,
                        'T_wall_max': 600 + 200 * (pct/100)
                    })

                plotter = ThrottleCurvePlotter()
                fig = plotter.create_figure(throttle_data)
                st.plotly_chart(fig, use_container_width=True)

                st.session_state.throttle_results = throttle_data
                st.success("Throttle map generated!")

            except Exception as e:
                st.error(f"Error generating throttle map: {e}")

    st.divider()

    st.subheader("Throttle Curve Data")
    if st.session_state.get('throttle_results'):
        import pandas as pd
        df = pd.DataFrame(st.session_state.throttle_results)
        st.dataframe(df)

        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "throttle_curve.csv")
