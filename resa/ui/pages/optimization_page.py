"""Optimization Page for RESA UI."""
import streamlit as st


def render_optimization_page():
    """Render the optimization page."""
    st.title("🎯 Multi-Point Optimization")

    st.subheader("Optimization Setup")

    st.markdown("### Design Variables")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.checkbox("Chamber Pressure", value=True)
        pc_min = st.number_input("Pc Min [bar]", 10.0, 50.0, 15.0)
        pc_max = st.number_input("Pc Max [bar]", 20.0, 60.0, 35.0)
    with col2:
        st.checkbox("Mixture Ratio", value=True)
        mr_min = st.number_input("MR Min", 2.0, 6.0, 3.0)
        mr_max = st.number_input("MR Max", 4.0, 8.0, 5.5)
    with col3:
        st.checkbox("L* (mm)", value=False)
        lstar_min = st.number_input("L* Min", 800.0, 1200.0, 900.0)
        lstar_max = st.number_input("L* Max", 1000.0, 1500.0, 1300.0)

    st.markdown("### Objective")
    objective = st.selectbox(
        "Optimize for",
        ["Maximize Isp", "Minimize Mass", "Maximize Thrust-to-Weight", "Minimize Wall Temperature"]
    )

    st.markdown("### Constraints")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Max Wall Temp [K]", 600.0, 1000.0, 800.0)
        st.number_input("Min Thrust [N]", 100.0, 5000.0, 1500.0)
    with col2:
        st.number_input("Max Pressure Drop [bar]", 5.0, 30.0, 15.0)
        st.number_input("Min Cooling Margin [K]", 0.0, 200.0, 50.0)

    st.markdown("### Algorithm")
    algorithm = st.selectbox("Optimizer", ["Nelder-Mead", "Powell", "SLSQP", "Differential Evolution"])
    max_iter = st.number_input("Max Iterations", 10, 500, 100)

    if st.button("🎯 Run Optimization", type="primary"):
        with st.spinner("Optimizing..."):
            st.info("Optimization would run here with selected parameters and constraints.")
            st.success("Optimization complete! (Demo mode)")
