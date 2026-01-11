"""
RESA - Rocket Engine Sizing & Analysis
Main Streamlit Application Entry Point

Run with: streamlit run app.py
"""
import streamlit as st
from pathlib import Path
import sys

from rocket_engine.ui.pages import render_design_page

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="RESA - Rocket Engine Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/resa',
        'Report a bug': 'https://github.com/your-repo/resa/issues',
        'About': "RESA v1.0 - Rocket Engine Sizing & Analysis Tool"
    }
)

# Custom CSS for better appearance
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Headers */
    h1 {
        color: #1a1a2e;
        font-weight: 700;
    }
    
    h2, h3 {
        color: #16213e;
        border-bottom: 2px solid #e94560;
        padding-bottom: 0.5rem;
    }
    
    /* Success/Warning boxes */
    .stSuccess, .stWarning, .stError {
        border-radius: 8px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
    
    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'engine_config': None,
        'design_result': None,
        'throttle_results': None,
        'analysis_history': [],
        'current_page': 'home',
        'dark_mode': False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render the navigation sidebar."""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80?text=RESA", width=200)
        st.markdown("---")
        
        st.markdown("### 🧭 Navigation")
        
        pages = {
            '🏠 Home': 'home',
            '🔧 Engine Design': 'design',
            '📊 Performance Analysis': 'analysis',
            '🌡️ Thermal Analysis': 'thermal',
            '💉 Injector Design': 'injector',
            '⚡ Throttle Curves': 'throttle',
            '🔬 Fluid Properties': 'fluids',
            '📁 Project Manager': 'projects',
        }
        
        for label, page_id in pages.items():
            if st.button(label, key=f"nav_{page_id}", use_container_width=True):
                st.session_state.current_page = page_id
                st.rerun()
        
        st.markdown("---")
        
        # Quick status panel
        if st.session_state.engine_config:
            cfg = st.session_state.engine_config
            st.markdown("### 📋 Active Engine")
            st.caption(f"**{cfg.engine_name}**")
            st.caption(f"Thrust: {cfg.thrust_n:.0f} N")
            st.caption(f"Pc: {cfg.pc_bar:.1f} bar")
            st.caption(f"Propellants: {cfg.fuel}/{cfg.oxidizer}")
        else:
            st.info("No engine loaded")
        
        st.markdown("---")
        st.caption("RESA v1.0 | © 2025")


def render_home_page():
    """Render the home/dashboard page."""
    st.title("🚀 RESA - Rocket Engine Sizing & Analysis")
    
    st.markdown("""
    Welcome to RESA, a comprehensive tool for liquid rocket engine preliminary design 
    and analysis. This tool integrates combustion analysis (CEA), regenerative cooling 
    simulation, and injector design capabilities.
    """)
    
    # Quick action cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔧 Quick Design")
        st.markdown("Start a new engine design with guided inputs.")
        if st.button("New Engine Design", type="primary", key="home_new_design"):
            st.session_state.current_page = 'design'
            st.rerun()
    
    with col2:
        st.markdown("### 📂 Load Project")
        st.markdown("Load an existing engine configuration.")
        uploaded = st.file_uploader("Upload YAML config", type=['yaml', 'yml'], 
                                   key="home_upload", label_visibility="collapsed")
        if uploaded:
            try:
                import yaml
                from rocket_engine.core.config import EngineConfig
                data = yaml.safe_load(uploaded)
                st.session_state.engine_config = EngineConfig._from_nested_dict(data)
                st.success(f"Loaded: {st.session_state.engine_config.engine_name}")
            except Exception as e:
                st.error(f"Error loading config: {e}")
    
    with col3:
        st.markdown("### 📚 Presets")
        st.markdown("Load a pre-configured engine template.")
        from rocket_engine.core.config import AnalysisPreset
        preset_choice = st.selectbox(
            "Select Preset",
            ["Demo 50N", "Hopper 2kN"],
            key="home_preset",
            label_visibility="collapsed"
        )
        if st.button("Load Preset", key="home_load_preset"):
            if preset_choice == "Demo 50N":
                preset = AnalysisPreset.demo_50n()
            else:
                preset = AnalysisPreset.hopper_2kn()
            st.session_state.engine_config = preset.config
            st.success(f"Loaded preset: {preset.name}")
            st.rerun()
    
    st.markdown("---")
    
    # Recent analysis history
    if st.session_state.analysis_history:
        st.markdown("### 📜 Recent Analyses")
        for i, entry in enumerate(st.session_state.analysis_history[-5:]):
            st.caption(f"{entry['timestamp']} - {entry['name']} - {entry['type']}")
    
    # Feature overview
    st.markdown("---")
    st.markdown("### ✨ Features")
    
    features = [
        ("🔥 Combustion Analysis", "CEA-based equilibrium chemistry for accurate performance prediction"),
        ("❄️ Regenerative Cooling", "1D marching solver with real fluid properties via CoolProp"),
        ("💉 Injector Design", "Swirl injector sizing with spray angle and Cd estimation"),
        ("📈 Throttle Analysis", "Operating envelope mapping for deep throttling"),
        ("🔬 Two-Phase Flow", "N2O orifice flow models (SPI, HEM, Dyer, FML)"),
        ("📊 Visualization", "Interactive Plotly charts and 3D geometry views"),
    ]
    
    cols = st.columns(2)
    for i, (title, desc) in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"**{title}**")
            st.caption(desc)


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    
    # Route to appropriate page
    page = st.session_state.current_page
    
    if page == 'home':
        render_home_page()
    elif page == 'design':
        render_design_page()
    elif page == 'analysis':
        from rocket_engine.ui.pages.analysis_page import render_analysis_page
        render_analysis_page()
    elif page == 'thermal':
        from rocket_engine.ui.pages.thermal_page import render_thermal_page
        render_thermal_page()
    elif page == 'injector':
        from rocket_engine.ui.pages.injector_page import render_injector_page
        render_injector_page()
    elif page == 'throttle':
        from rocket_engine.ui.pages.throttle_page import render_throttle_page
        render_throttle_page()
    elif page == 'fluids':
        from rocket_engine.ui.pages.fluids_page import render_fluids_page
        render_fluids_page()
    elif page == 'projects':
        from rocket_engine.ui.pages.projects_page import render_projects_page
        render_projects_page()
    else:
        st.error(f"Unknown page: {page}")


if __name__ == "__main__":
    main()
