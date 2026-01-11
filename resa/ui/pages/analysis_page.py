"""Analysis Page for RESA UI."""
import streamlit as st


def render_analysis_page():
    """Render the analysis page."""
    st.title("📊 Performance Analysis")

    if not st.session_state.get('design_result'):
        st.warning("Please run an engine design first.")
        return

    result = st.session_state.design_result

    st.subheader("Off-Design Analysis")

    col1, col2 = st.columns(2)
    with col1:
        new_pc = st.slider(
            "Chamber Pressure [bar]",
            min_value=5.0,
            max_value=50.0,
            value=float(result.pc_bar)
        )
    with col2:
        new_mr = st.slider(
            "Mixture Ratio",
            min_value=1.0,
            max_value=8.0,
            value=float(result.mr)
        )

    if st.button("Run Off-Design Analysis"):
        st.info("Off-design analysis would run here with modified Pc and MR")

    st.divider()

    st.subheader("Comparison")
    st.info("Compare multiple design points by running analyses and storing results.")
