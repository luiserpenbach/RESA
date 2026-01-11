"""
RESA - Rocket Engine Sizing & Analysis

A professional toolkit for liquid rocket engine design, analysis, and visualization.
"""

__version__ = "2.0.0"

# Core exports
from resa.core.interfaces import Solver, GeometryGenerator, FluidProvider, Plotter, ReportGenerator
from resa.core.engine import LiquidEngine
from resa.core.results import EngineDesignResult
from resa.config.engine_config import EngineConfig

# Visualization exports
from resa.visualization.engine_plots import EngineDashboardPlotter, CrossSectionPlotter
from resa.visualization.themes import PlotTheme, EngineeringTheme

# Reporting exports
from resa.reporting.html_report import HTMLReportGenerator

__all__ = [
    # Core
    "LiquidEngine",
    "EngineConfig",
    "EngineDesignResult",
    # Interfaces
    "Solver",
    "GeometryGenerator",
    "FluidProvider",
    "Plotter",
    "ReportGenerator",
    # Visualization
    "EngineDashboardPlotter",
    "CrossSectionPlotter",
    "PlotTheme",
    "EngineeringTheme",
    # Reporting
    "HTMLReportGenerator",
]
