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
    page_title="RESA — Rocket Engine Design Suite",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global dark-modern CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & body ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: #e0e0e0;
}

.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1440px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141a26 0%, #0b0f1a 100%);
    border-right: 1px solid #1f2d45;
}

[data-testid="stSidebar"] * {
    color: #c8d6e5 !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: transparent;
    border: 1px solid #1f2d45;
    color: #c8d6e5 !important;
    text-align: left;
    border-radius: 6px;
    font-size: 0.85rem;
    padding: 0.45rem 0.9rem;
    transition: all 0.2s ease;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #1a2744;
    border-color: #2e6fff;
    color: #ffffff !important;
}

[data-testid="stSidebar"] hr {
    border-color: #1f2d45;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d2137 0%, #0b1a2e 100%);
    border: 1px solid #1a3a5c;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}

[data-testid="metric-container"] label {
    color: #7ba7cc !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8f4fd !important;
    font-size: 1.55rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
}

/* ── Headings ── */
h1 { color: #e8f4fd; font-weight: 700; letter-spacing: -0.02em; }
h2 { color: #c8d6e5; font-weight: 600; }
h3 { color: #a8bfd6; font-weight: 600; }

/* ── Divider ── */
hr { border-color: #1f2d45 !important; }

/* ── Buttons (primary) ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2e6fff 0%, #1a3fa8 100%);
    border: none;
    color: #ffffff !important;
    border-radius: 8px;
    padding: 0.55rem 1.6rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4a82ff 0%, #2255cc 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(46, 111, 255, 0.35);
}

/* ── Buttons (secondary / default) ── */
.stButton > button {
    background: #111827;
    border: 1px solid #2a3f5c;
    color: #c8d6e5 !important;
    border-radius: 8px;
    padding: 0.5rem 1.4rem;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #2e6fff;
    background: #162035;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #111827;
    border-radius: 10px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    padding: 8px 18px;
    color: #7ba7cc !important;
    font-size: 0.85rem;
    font-weight: 500;
    background: transparent !important;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a3a6e 0%, #0d2650 100%) !important;
    color: #e8f4fd !important;
    border: 1px solid #2255cc !important;
}

/* ── Alerts ── */
.stSuccess {
    background-color: #0a2618 !important;
    border-left: 3px solid #21c97a !important;
    color: #7cf4b6 !important;
    border-radius: 6px;
}

.stWarning {
    background-color: #231a06 !important;
    border-left: 3px solid #f0a000 !important;
    color: #f5d080 !important;
    border-radius: 6px;
}

.stError {
    background-color: #1e0a0a !important;
    border-left: 3px solid #d64045 !important;
    color: #f08080 !important;
    border-radius: 6px;
}

.stInfo {
    background-color: #091a2e !important;
    border-left: 3px solid #2e6fff !important;
    color: #7bbfff !important;
    border-radius: 6px;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div,
[data-testid="stTextArea"] textarea {
    background: #111827 !important;
    border-color: #2a3f5c !important;
    color: #e0e0e0 !important;
    border-radius: 6px;
}

/* ── Sliders ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #2e6fff !important;
    border-color: #4a82ff !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #111827;
    border: 1px solid #1f2d45;
    border-radius: 8px;
}

[data-testid="stExpander"] summary {
    color: #7ba7cc !important;
    font-weight: 600;
}

/* ── Feature card (HTML) ── */
.resa-card {
    background: linear-gradient(135deg, #111827 0%, #0d1520 100%);
    border: 1px solid #1f2d45;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s ease;
}

.resa-card:hover {
    border-color: #2e6fff;
}

.resa-card h4 {
    color: #e8f4fd;
    margin: 0 0 0.5rem 0;
    font-size: 1rem;
    font-weight: 600;
}

.resa-card p {
    color: #7ba7cc;
    margin: 0;
    font-size: 0.85rem;
    line-height: 1.55;
}

/* ── Status badge ── */
.badge-safe {
    display: inline-block;
    background: #0a2618;
    color: #21c97a;
    border: 1px solid #21c97a;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}

.badge-warn {
    display: inline-block;
    background: #231a06;
    color: #f0a000;
    border: 1px solid #f0a000;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}

.badge-danger {
    display: inline-block;
    background: #1e0a0a;
    color: #d64045;
    border: 1px solid #d64045;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1f2d45;
    border-radius: 8px;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #2e6fff !important;
}

/* ── Caption / small text ── */
.stCaption, small, caption {
    color: #4a6a8a !important;
}

/* ── Hide Streamlit auto-generated page nav ── */
[data-testid="stSidebarNav"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)


# ─── Navigation groups ────────────────────────────────────────────────────
_NAV_GROUPS = [
    ("Thrust Chamber", [
        ("home",     "Dashboard"),
        ("design",   "Engine Design"),
        ("cooling",  "Cooling Analysis"),
        ("contour",  "3D Contour"),
    ]),
    ("Performance Analysis", [
        ("analysis",     "Off-Design Analysis"),
        ("throttle",     "Throttle Map"),
        ("monte_carlo",  "Monte Carlo"),
        ("optimization", "Optimization"),
    ]),
    ("Component Plug-ins", [
        ("injector", "Injector Design"),
        ("igniter",  "Igniter Design"),
        ("tank",     "Tank Simulation"),
    ]),
    ("Settings", [
        ("projects", "Projects"),
        ("settings", "Settings"),
    ]),
]


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "current_page": "home",
        "engine_config": None,
        "design_result": None,
        "throttle_results": None,
        "monte_carlo_results": None,
        "project_name": None,
        "analysis_history": [],
        "theme": "dark",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _nav_group_header(label: str):
    """Render a small all-caps section label for nav groups."""
    st.markdown(
        f'<div style="font-size:0.68rem;color:#4a6a8a;text-transform:uppercase;'
        f'letter-spacing:0.1em;padding:0.8rem 0.6rem 0.2rem;font-weight:600;">'
        f"{label}</div>",
        unsafe_allow_html=True,
    )


def _nav_button(page_id: str, label: str):
    """Render a single sidebar nav button."""
    active = st.session_state.current_page == page_id
    if active:
        st.markdown(
            f'<div style="background:#1a3a6e;border:1px solid #2255cc;'
            f'border-radius:6px;padding:6px 12px;margin-bottom:4px;'
            f'color:#e8f4fd;font-size:0.85rem;font-weight:600;">'
            f"{label}</div>",
            unsafe_allow_html=True,
        )
    else:
        if st.button(label, key=f"nav_{page_id}", use_container_width=True):
            st.session_state.current_page = page_id
            st.rerun()


def render_sidebar():
    """Render the navigation sidebar."""
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:0.5rem 0 0.3rem;">'
            '<div style="font-size:1.5rem;font-weight:800;color:#e8f4fd;'
            'letter-spacing:-0.02em;">RESA</div>'
            '<div style="font-size:0.72rem;color:#4a6a8a;letter-spacing:0.12em;'
            'text-transform:uppercase;">Rocket Engine Suite</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        for group_label, pages in _NAV_GROUPS:
            _nav_group_header(group_label)
            for page_id, label in pages:
                _nav_button(page_id, label)

        st.divider()

        # Active engine status
        if st.session_state.engine_config:
            cfg = st.session_state.engine_config
            st.markdown(
                f'<div class="resa-card">'
                f"<h4>{cfg.engine_name}</h4>"
                f"<p>Thrust: {cfg.thrust_n:.0f} N &nbsp;|&nbsp; Pc: {cfg.pc_bar:.1f} bar</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("New", use_container_width=True):
                st.session_state.engine_config = None
                st.session_state.design_result = None
                st.session_state.current_page = "design"
                st.rerun()
        with col2:
            if st.button("Save", use_container_width=True):
                st.session_state.current_page = "projects"
                st.rerun()

        st.divider()
        st.caption("RESA v2.0.0  •  MIT License")


# ─── Home dashboard ────────────────────────────────────────────────────────
def render_home_page():
    st.markdown(
        '<h1 style="font-size:2rem;">RESA Dashboard</h1>'
        '<p style="color:#4a6a8a;margin-top:-0.5rem;">'
        'Rocket Engine Sizing &amp; Analysis &mdash; v2.0.0</p>',
        unsafe_allow_html=True,
    )

    # Quick-stats row
    col1, col2, col3, col4 = st.columns(4)
    dr = st.session_state.design_result

    with col1:
        st.metric("Isp (vac)", f"{dr.isp_vac:.1f} s" if dr else "—")
    with col2:
        st.metric("Thrust", f"{dr.thrust_vac:.0f} N" if dr else "—")
    with col3:
        st.metric("Mass Flow", f"{dr.massflow_total:.3f} kg/s" if dr else "—")
    with col4:
        if dr and getattr(dr, "cooling", None):
            st.metric("Max T_wall", f"{dr.cooling.max_wall_temp:.0f} K")
        else:
            st.metric("Max T_wall", "—")

    st.divider()

    # Feature cards
    st.markdown("### Modules")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            '<div class="resa-card"><h4>Cooling Analysis</h4>'
            "<p>Two-phase N2O regenerative cooling with CHF tracking, "
            "Bartz heat flux, Chen boiling, supercritical correlations, "
            "and parametric sweeps.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Open Cooling Analysis", key="h_cooling"):
            st.session_state.current_page = "cooling"
            st.rerun()

    with c2:
        st.markdown(
            '<div class="resa-card"><h4>Engine Design</h4>'
            "<p>Full engine sizing: CEA combustion, bell nozzle contour, "
            "regenerative cooling, performance dashboard and HTML reports.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Open Engine Design", key="h_design"):
            st.session_state.current_page = "design"
            st.rerun()

    with c3:
        st.markdown(
            '<div class="resa-card"><h4>Monte Carlo</h4>'
            "<p>Latin Hypercube Sampling uncertainty quantification "
            "with sensitivity tornado charts and statistical distributions.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Open Monte Carlo", key="h_mc"):
            st.session_state.current_page = "monte_carlo"
            st.rerun()

    st.divider()

    # Recent activity
    st.markdown("### Recent Activity")
    if st.session_state.analysis_history:
        for item in reversed(st.session_state.analysis_history[-8:]):
            st.markdown(f"<small style='color:#4a6a8a;'>• {item}</small>", unsafe_allow_html=True)
    else:
        st.info("No recent activity. Start by designing a new engine or running a cooling analysis.")


# ─── Page loader stubs ─────────────────────────────────────────────────────
def render_cooling_page():
    from resa.ui.pages.cooling_page import render_cooling_page as _render
    _render()


def render_design_page():
    from resa.ui.pages.design_page import render_design_page as _render
    _render()


def render_analysis_page():
    from resa.ui.pages.analysis_page import render_analysis_page as _render
    _render()


def render_throttle_page():
    from resa.ui.pages.throttle_page import render_throttle_page as _render
    _render()


def render_monte_carlo_page():
    from resa.ui.pages.monte_carlo_page import render_monte_carlo_page as _render
    _render()


def render_optimization_page():
    from resa.ui.pages.optimization_page import render_optimization_page as _render
    _render()


def render_injector_page():
    from resa.ui.pages.injector_page import render_injector_page as _render
    _render()


def render_igniter_page():
    from resa.ui.pages.igniter_page import render_igniter_page as _render
    _render()


def render_contour_page():
    from resa.ui.pages.contour_page import render_contour_page as _render
    _render()


def render_tank_page():
    from resa.ui.pages.tank_page import render_tank_page as _render
    _render()


def render_projects_page():
    from resa.ui.pages.projects_page import render_projects_page as _render
    _render()


def render_settings_page():
    st.markdown("<h1>Settings</h1>", unsafe_allow_html=True)

    st.subheader("Output Directory")
    st.text_input("Output directory", value="./output")

    st.subheader("Units")
    st.selectbox("Unit system", ["SI", "Imperial", "Mixed"])

    if st.button("Save Settings", type="primary"):
        st.success("Settings saved!")


# ─── Router ────────────────────────────────────────────────────────────────
_PAGE_ROUTES = {
    "home":         render_home_page,
    "cooling":      render_cooling_page,
    "design":       render_design_page,
    "analysis":     render_analysis_page,
    "throttle":     render_throttle_page,
    "monte_carlo":  render_monte_carlo_page,
    "optimization": render_optimization_page,
    "injector":     render_injector_page,
    "igniter":      render_igniter_page,
    "contour":      render_contour_page,
    "tank":         render_tank_page,
    "projects":     render_projects_page,
    "settings":     render_settings_page,
}


def main():
    init_session_state()
    render_sidebar()

    page = st.session_state.current_page
    handler = _PAGE_ROUTES.get(page, render_home_page)
    try:
        handler()
    except ImportError as exc:
        st.error(f"Page module not available: {exc}")
        st.info("This module is under development.")


if __name__ == "__main__":
    main()
