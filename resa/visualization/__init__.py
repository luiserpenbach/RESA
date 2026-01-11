"""
RESA Visualization Module

All visualizations are built on Plotly for:
- Interactive charts (zoom, hover, pan)
- Easy HTML embedding for reports
- Consistent theming
- Streamlit compatibility
- WebGL-accelerated 3D graphics
"""

from resa.visualization.themes import PlotTheme, EngineeringTheme, DarkTheme
from resa.visualization.engine_plots import EngineDashboardPlotter, CrossSectionPlotter
from resa.visualization.performance_plots import ThrottleCurvePlotter, PerformanceContourPlotter
from resa.visualization.engine_3d import (
    Engine3DViewer,
    quick_nozzle_3d,
    quick_channels_3d,
    quick_engine_3d,
)

__all__ = [
    # Themes
    "PlotTheme",
    "EngineeringTheme",
    "DarkTheme",
    # 2D Plotters
    "EngineDashboardPlotter",
    "CrossSectionPlotter",
    "ThrottleCurvePlotter",
    "PerformanceContourPlotter",
    # 3D Visualization
    "Engine3DViewer",
    "quick_nozzle_3d",
    "quick_channels_3d",
    "quick_engine_3d",
]
