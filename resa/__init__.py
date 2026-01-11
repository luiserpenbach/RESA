"""
RESA - Rocket Engine Sizing & Analysis
=======================================

A state-of-the-art Python toolkit for liquid rocket engine preliminary design,
analysis, and optimization. Features include combustion analysis, regenerative
cooling, injector design, 3D visualization, and Monte Carlo uncertainty analysis.

Package Structure:
------------------
- core: Configuration, results, interfaces, and exceptions
- physics: Pure physics calculations (combustion, heat transfer, fluid flow)
- geometry: Nozzle and cooling channel geometry generation
- solvers: Integrated analysis solvers (CEA, cooling, flow)
- addons: Specialized design tools
    - igniter: Torch igniter sizing and analysis
    - injector: Swirl injector (LCSC/GCSC) design
    - contour: Advanced 3D nozzle contour generation
    - tank: Tank pressurization and depletion simulation
- visualization: Plotly-based interactive visualizations
- reporting: HTML/PDF report generation
- analysis: Monte Carlo, optimization, and sensitivity analysis
- ui: Streamlit-based graphical interface
- utils: Utility functions and constants
- projects: Version-controlled project management

Quick Start:
------------
>>> from resa import Engine, EngineConfig
>>>
>>> config = EngineConfig(
...     engine_name="Phoenix-1",
...     fuel="Ethanol90",
...     oxidizer="N2O",
...     thrust_n=2200,
...     pc_bar=25,
...     mr=4.0
... )
>>>
>>> engine = Engine(config)
>>> result = engine.design()
>>> result.to_html("phoenix1_report.html")
>>>
>>> # Run with Streamlit UI:
>>> # streamlit run -m resa.ui.app

Features:
---------
- NASA CEA-based combustion analysis
- Regenerative cooling with real fluid properties (CoolProp)
- Bell and conical nozzle contour generation
- LCSC/GCSC swirl injector sizing
- Torch igniter design with HEM two-phase flow
- Interactive Plotly visualizations
- 3D WebGL nozzle viewer
- Monte Carlo uncertainty analysis
- Multi-point throttle optimization
- Git-integrated version control for designs
- Professional HTML report generation

Author: RESA Development Team
License: MIT
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "RESA Development Team"

# Core imports
from resa.core.config import EngineConfig
from resa.core.results import EngineDesignResult, CombustionResult
from resa.core.interfaces import Solver, FluidProvider, GeometryGenerator
from resa.core.exceptions import RESAError, ConfigurationError, PhysicsError

# Main engine class
from resa.core.engine import Engine

# Visualization
from resa.visualization.themes import PlotTheme, EngineeringTheme, DarkTheme
from resa.visualization.engine_plots import EngineDashboardPlotter, CrossSectionPlotter
from resa.visualization.engine_3d import Engine3DViewer

# Reporting
from resa.reporting.html_report import HTMLReportGenerator

# Analysis
from resa.analysis.monte_carlo import MonteCarloAnalysis
from resa.analysis.optimization import ThrottleOptimizer

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Core
    "Engine",
    "EngineConfig",
    "EngineDesignResult",
    "CombustionResult",
    # Interfaces
    "Solver",
    "FluidProvider",
    "GeometryGenerator",
    # Exceptions
    "RESAError",
    "ConfigurationError",
    "PhysicsError",
    # Visualization
    "PlotTheme",
    "EngineeringTheme",
    "DarkTheme",
    "EngineDashboardPlotter",
    "CrossSectionPlotter",
    "Engine3DViewer",
    # Reporting
    "HTMLReportGenerator",
    # Analysis
    "MonteCarloAnalysis",
    "ThrottleOptimizer",
]
