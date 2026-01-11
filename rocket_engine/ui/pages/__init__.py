"""
RESA UI Pages
"""
from .design_page import render_design_page
from .analysis_page import render_analysis_page
from .thermal_page import render_thermal_page
from .injector_page import render_injector_page
from .throttle_page import render_throttle_page
from .fluids_page import render_fluids_page
from .projects_page import render_projects_page

__all__ = [
    'render_design_page',
    'render_analysis_page',
    'render_thermal_page',
    'render_injector_page',
    'render_throttle_page',
    'render_fluids_page',
    'render_projects_page',
]
