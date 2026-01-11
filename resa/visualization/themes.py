"""
Centralized theming for all RESA visualizations.

Provides consistent colors, fonts, and styling across:
- Engine dashboard plots
- Cross-section views
- Performance contours
- HTML reports
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go


@dataclass
class PlotTheme:
    """
    Base theme configuration for all plots.

    Attributes:
        primary: Main color for primary data series
        secondary: Color for secondary data series
        accent: Highlight color
        danger: Color for warnings/high values (temperature)
        background: Plot background color
        grid_color: Color for grid lines
    """

    # Core colors
    primary: str = "#1f77b4"
    secondary: str = "#ff7f0e"
    accent: str = "#2ca02c"
    danger: str = "#d62728"
    info: str = "#17becf"

    # Background colors
    background: str = "#ffffff"
    paper_background: str = "#ffffff"
    grid_color: str = "#e5e5e5"

    # Material colors (for engine components)
    copper: str = "#B87333"
    coolant: str = "#4db6ac"
    gas: str = "#ffccbc"
    steel: str = "#A9A9A9"
    nickel: str = "#8B8B8B"

    # Typography
    font_family: str = "Arial, Helvetica, sans-serif"
    title_size: int = 18
    axis_title_size: int = 14
    tick_size: int = 12
    annotation_size: int = 11

    # Layout
    line_width: int = 2
    marker_size: int = 8
    margin: Dict[str, int] = field(default_factory=lambda: {
        "l": 70, "r": 70, "t": 90, "b": 70
    })

    # Color scales for heatmaps
    temperature_colorscale: List = field(default_factory=lambda: [
        [0.0, "#3498db"],   # Cool blue
        [0.3, "#2ecc71"],   # Green
        [0.5, "#f1c40f"],   # Yellow
        [0.7, "#e67e22"],   # Orange
        [1.0, "#e74c3c"],   # Hot red
    ])

    pressure_colorscale: List = field(default_factory=lambda: [
        [0.0, "#ecf0f1"],   # Light gray
        [0.5, "#3498db"],   # Blue
        [1.0, "#2c3e50"],   # Dark blue
    ])

    def apply_to_figure(self, fig: go.Figure) -> go.Figure:
        """
        Apply this theme to a Plotly figure.

        Args:
            fig: Plotly Figure to style

        Returns:
            The styled figure (modified in place)
        """
        fig.update_layout(
            font=dict(
                family=self.font_family,
                size=self.tick_size,
            ),
            title=dict(
                font=dict(size=self.title_size),
                x=0.5,
                xanchor='center',
            ),
            paper_bgcolor=self.paper_background,
            plot_bgcolor=self.background,
            margin=self.margin,
        )

        # Update axes
        fig.update_xaxes(
            title_font=dict(size=self.axis_title_size),
            tickfont=dict(size=self.tick_size),
            gridcolor=self.grid_color,
            linecolor=self.grid_color,
            showgrid=True,
            gridwidth=1,
        )

        fig.update_yaxes(
            title_font=dict(size=self.axis_title_size),
            tickfont=dict(size=self.tick_size),
            gridcolor=self.grid_color,
            linecolor=self.grid_color,
            showgrid=True,
            gridwidth=1,
        )

        return fig

    def get_color_sequence(self) -> List[str]:
        """Get ordered color sequence for multi-series plots."""
        return [
            self.primary,
            self.secondary,
            self.accent,
            self.danger,
            self.info,
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
        ]

    def to_plotly_template(self) -> Dict[str, Any]:
        """Convert theme to Plotly template dictionary."""
        return {
            "layout": {
                "colorway": self.get_color_sequence(),
                "font": {"family": self.font_family},
            }
        }


@dataclass
class EngineeringTheme(PlotTheme):
    """
    Professional engineering-focused theme.

    Optimized for:
    - Technical reports
    - Print-friendly colors
    - Clear data visualization
    """

    # More muted, professional colors
    primary: str = "#2c3e50"
    secondary: str = "#e74c3c"
    accent: str = "#27ae60"
    danger: str = "#c0392b"
    info: str = "#2980b9"

    # Lighter background for print
    background: str = "#fafafa"
    paper_background: str = "#ffffff"
    grid_color: str = "#ddd"

    # Slightly larger fonts for readability
    title_size: int = 20
    axis_title_size: int = 14

    def apply_to_figure(self, fig: go.Figure) -> go.Figure:
        """Apply engineering theme with additional professional styling."""
        super().apply_to_figure(fig)

        # Add subtle border
        fig.update_layout(
            xaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
            ),
            yaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
            ),
        )

        return fig


@dataclass
class DarkTheme(PlotTheme):
    """
    Dark theme for presentations and UI.

    Optimized for:
    - Dark mode applications
    - Presentations
    - Streamlit dark theme
    """

    primary: str = "#3498db"
    secondary: str = "#e74c3c"
    accent: str = "#2ecc71"
    danger: str = "#e74c3c"
    info: str = "#1abc9c"

    background: str = "#1e1e1e"
    paper_background: str = "#252525"
    grid_color: str = "#404040"

    # Brighter materials for visibility
    copper: str = "#cd7f32"
    coolant: str = "#5dade2"
    steel: str = "#bdc3c7"

    def apply_to_figure(self, fig: go.Figure) -> go.Figure:
        """Apply dark theme."""
        super().apply_to_figure(fig)

        fig.update_layout(
            font=dict(color="#ffffff"),
        )

        fig.update_xaxes(
            title_font=dict(color="#ffffff"),
            tickfont=dict(color="#cccccc"),
        )

        fig.update_yaxes(
            title_font=dict(color="#ffffff"),
            tickfont=dict(color="#cccccc"),
        )

        return fig


# Default theme instance
DEFAULT_THEME = EngineeringTheme()
