"""
RESA - Rocket Engine Sizing & Analysis
Unified Streamlit UI Application

Run with: streamlit run resa/ui/app.py
"""
import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

# Page configuration must be first Streamlit command
st.set_page_config(
    page_title="RESA - Rocket Engine Design Suite",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }

    [data-testid="metric-container"] label {
        color: rgba(255,255,255,0.8) !important;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: white !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #0d1b2a 100%);
    }

    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
    }

    /* Headers */
    h1, h2, h3 {
        color: #1e3a5f;
    }

    /* Cards */
    .stCard {
        border-radius: 10px;
        padding: 1.5rem;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }

    /* Success/Warning boxes */
    .stSuccess, .stWarning, .stError {
        border-radius: 8px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'current_page': 'home',
        'engine_config': None,
        'design_result': None,
        'throttle_results': None,
        'monte_carlo_results': None,
        'project_name': None,
        'analysis_history': [],
        'theme': 'light'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render the navigation sidebar."""
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/rocket.svg", width=50)
        st.title("RESA")
        st.caption("Rocket Engine Sizing & Analysis")

        st.divider()

        # Navigation
        st.subheader("Navigation")

        pages = {
            'home': ('🏠', 'Dashboard'),
            'design': ('🔧', 'Engine Design'),
            'analysis': ('📊', 'Analysis'),
            'throttle': ('⚡', 'Throttle Map'),
            'monte_carlo': ('🎲', 'Monte Carlo'),
            'optimization': ('🎯', 'Optimization'),
            'injector': ('💧', 'Injector Design'),
            'igniter': ('🔥', 'Igniter Design'),
            'contour': ('📐', '3D Contour'),
            'tank': ('🛢️', 'Tank Simulation'),
            'projects': ('📁', 'Projects'),
            'settings': ('⚙️', 'Settings'),
        }

        for page_id, (icon, name) in pages.items():
            if st.button(f"{icon} {name}", key=f"nav_{page_id}", use_container_width=True):
                st.session_state.current_page = page_id
                st.rerun()

        st.divider()

        # Active engine info
        if st.session_state.engine_config:
            st.subheader("Active Engine")
            cfg = st.session_state.engine_config
            st.info(f"**{cfg.engine_name}**\n\n"
                   f"Thrust: {cfg.thrust_n:.0f} N\n\n"
                   f"Pc: {cfg.pc_bar:.1f} bar")

        # Quick actions
        st.subheader("Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 New", use_container_width=True):
                st.session_state.engine_config = None
                st.session_state.design_result = None
                st.session_state.current_page = 'design'
                st.rerun()
        with col2:
            if st.button("💾 Save", use_container_width=True):
                st.session_state.current_page = 'projects'
                st.rerun()

        st.divider()
        st.caption("RESA v2.0.0")


def render_home_page():
    """Render the home dashboard."""
    st.title("🚀 RESA Dashboard")
    st.markdown("**Rocket Engine Sizing & Analysis** - State-of-the-art design tool")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.session_state.design_result:
            st.metric("Isp (vac)", f"{st.session_state.design_result.isp_vac:.1f} s")
        else:
            st.metric("Isp (vac)", "---")

    with col2:
        if st.session_state.design_result:
            st.metric("Thrust", f"{st.session_state.design_result.thrust_vac:.0f} N")
        else:
            st.metric("Thrust", "---")

    with col3:
        if st.session_state.design_result:
            st.metric("Mass Flow", f"{st.session_state.design_result.massflow_total:.3f} kg/s")
        else:
            st.metric("Mass Flow", "---")

    with col4:
        if st.session_state.design_result and hasattr(st.session_state.design_result, 'cooling'):
            cooling = st.session_state.design_result.cooling
            if cooling:
                st.metric("Max T_wall", f"{cooling.max_wall_temp:.0f} K")
            else:
                st.metric("Max T_wall", "---")
        else:
            st.metric("Max T_wall", "---")

    st.divider()

    # Feature cards
    st.subheader("Getting Started")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔧 Engine Design
        Complete engine sizing with:
        - CEA combustion analysis
        - Bell nozzle contours
        - Regen cooling analysis

        [Start Design →](#)
        """)
        if st.button("Open Engine Design", key="home_design"):
            st.session_state.current_page = 'design'
            st.rerun()

    with col2:
        st.markdown("""
        ### 🎲 Monte Carlo
        Uncertainty analysis:
        - Parameter distributions
        - Sensitivity analysis
        - Statistical outputs

        [Run Analysis →](#)
        """)
        if st.button("Open Monte Carlo", key="home_mc"):
            st.session_state.current_page = 'monte_carlo'
            st.rerun()

    with col3:
        st.markdown("""
        ### 📐 3D Contour
        Advanced geometry:
        - STL export
        - Helical channels
        - CAD-ready output

        [Generate 3D →](#)
        """)
        if st.button("Open 3D Contour", key="home_contour"):
            st.session_state.current_page = 'contour'
            st.rerun()

    st.divider()

    # Recent activity
    st.subheader("Recent Activity")
    if st.session_state.analysis_history:
        for item in st.session_state.analysis_history[-5:]:
            st.text(f"• {item}")
    else:
        st.info("No recent activity. Start by creating a new engine design.")


def render_design_page():
    """Render the engine design page."""
    from resa.ui.pages.design_page import render_design_page as render
    render()


def render_analysis_page():
    """Render the analysis page."""
    from resa.ui.pages.analysis_page import render_analysis_page as render
    render()


def render_throttle_page():
    """Render the throttle mapping page."""
    from resa.ui.pages.throttle_page import render_throttle_page as render
    render()


def render_monte_carlo_page():
    """Render the Monte Carlo analysis page."""
    from resa.ui.pages.monte_carlo_page import render_monte_carlo_page as render
    render()


def render_optimization_page():
    """Render the optimization page."""
    from resa.ui.pages.optimization_page import render_optimization_page as render
    render()


def render_injector_page():
    """Render the injector design page."""
    from resa.ui.pages.injector_page import render_injector_page as render
    render()


def render_igniter_page():
    """Render the igniter design page."""
    from resa.ui.pages.igniter_page import render_igniter_page as render
    render()


def render_contour_page():
    """Render the 3D contour page."""
    from resa.ui.pages.contour_page import render_contour_page as render
    render()


def render_tank_page():
    """Render the tank simulation page."""
    from resa.ui.pages.tank_page import render_tank_page as render
    render()


def render_projects_page():
    """Render the projects page."""
    from resa.ui.pages.projects_page import render_projects_page as render
    render()


def render_settings_page():
    """Render the settings page."""
    st.title("⚙️ Settings")

    st.subheader("Output Directory")
    output_dir = st.text_input("Output directory", value="./output")

    st.subheader("Theme")
    theme = st.selectbox("Color theme", ["Light", "Dark", "Engineering"])

    st.subheader("Units")
    units = st.selectbox("Unit system", ["SI", "Imperial", "Mixed"])

    if st.button("Save Settings"):
        st.success("Settings saved!")


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()

    # Route to current page
    page_routes = {
        'home': render_home_page,
        'design': render_design_page,
        'analysis': render_analysis_page,
        'throttle': render_throttle_page,
        'monte_carlo': render_monte_carlo_page,
        'optimization': render_optimization_page,
        'injector': render_injector_page,
        'igniter': render_igniter_page,
        'contour': render_contour_page,
        'tank': render_tank_page,
        'projects': render_projects_page,
        'settings': render_settings_page,
    }

    current_page = st.session_state.current_page
    if current_page in page_routes:
        try:
            page_routes[current_page]()
        except ImportError as e:
            st.error(f"Page module not found: {e}")
            st.info("This page is under development.")
    else:
        render_home_page()


if __name__ == "__main__":
    main()
