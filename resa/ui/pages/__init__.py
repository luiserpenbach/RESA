"""UI Pages for RESA Streamlit application."""

from resa.ui.pages.design_page import render_design_page
from resa.ui.pages.analysis_page import render_analysis_page
from resa.ui.pages.throttle_page import render_throttle_page
from resa.ui.pages.monte_carlo_page import render_monte_carlo_page
from resa.ui.pages.optimization_page import render_optimization_page
from resa.ui.pages.injector_page import render_injector_page
from resa.ui.pages.igniter_page import render_igniter_page
from resa.ui.pages.contour_page import render_contour_page
from resa.ui.pages.tank_page import render_tank_page
from resa.ui.pages.projects_page import render_projects_page

__all__ = [
    "render_design_page",
    "render_analysis_page",
    "render_throttle_page",
    "render_monte_carlo_page",
    "render_optimization_page",
    "render_injector_page",
    "render_igniter_page",
    "render_contour_page",
    "render_tank_page",
    "render_projects_page",
]
