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

    # Additional material colors
    inconel: str = "#7B7B7B"
    aluminum: str = "#C0C0C0"
    graphite: str = "#4A4A4A"
    ceramic: str = "#F5F5DC"

    # Propellant colors
    oxidizer: str = "#00CED1"
    fuel: str = "#CD853F"
    combustion_gas: str = "#FF6347"

    # Flow visualization colors
    hot_gas: str = "#FF4500"
    cold_flow: str = "#1E90FF"
    spray_cone: str = "rgba(65, 105, 225, 0.3)"

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

    Matches the RESA dark-modern Streamlit aesthetic (#0e1117 base).
    Optimized for:
    - Dark mode Streamlit applications
    - Presentations and dashboards
    - High contrast data visualisation
    """

    # Core series colors — bright enough on dark bg
    primary: str = "#4a9eff"       # bright blue
    secondary: str = "#ff6b6b"     # coral red
    accent: str = "#2ecc71"        # emerald green
    danger: str = "#ff4d4d"        # vivid red
    info: str = "#1abc9c"          # teal

    # Backgrounds — match Streamlit dark
    background: str = "#0e1117"
    paper_background: str = "#111827"
    grid_color: str = "#1f2d45"

    # Font colors
    font_family: str = "Inter, Arial, Helvetica, sans-serif"
    title_size: int = 17
    axis_title_size: int = 13
    tick_size: int = 11

    # Material colors — vivid for dark bg
    copper: str = "#e89c4f"
    coolant: str = "#5dade2"
    gas: str = "#ff8c69"
    steel: str = "#c0c8d4"
    nickel: str = "#a0aab8"
    inconel: str = "#8898aa"
    aluminum: str = "#d0d8e4"
    graphite: str = "#7a8898"
    ceramic: str = "#d4cfa8"

    # Propellant colors
    oxidizer: str = "#00e5ff"
    fuel: str = "#ff9d3a"
    combustion_gas: str = "#ff5722"

    # Flow colours
    hot_gas: str = "#ff5722"
    cold_flow: str = "#2196f3"
    spray_cone: str = "rgba(65, 155, 255, 0.25)"

    # Color scales for heatmaps
    temperature_colorscale: List = field(default_factory=lambda: [
        [0.0, "#1a3a6e"],
        [0.25, "#2196f3"],
        [0.5, "#00e5ff"],
        [0.7, "#ffeb3b"],
        [0.85, "#ff9800"],
        [1.0, "#ff1744"],
    ])

    pressure_colorscale: List = field(default_factory=lambda: [
        [0.0, "#111827"],
        [0.5, "#1a5276"],
        [1.0, "#4a9eff"],
    ])

    def apply_to_figure(self, fig: go.Figure) -> go.Figure:
        """Apply dark theme with full layout and axis styling."""
        # Use Plotly's built-in dark template as the base so that trace colour
        # defaults, hover labels, and modebar all use dark-mode-appropriate values
        fig.update_layout(template="plotly_dark")

        # Override with RESA's custom dark settings
        fig.update_layout(
            font=dict(
                family=self.font_family,
                size=self.tick_size,
                color="#c8d6e5",
            ),
            title=dict(
                font=dict(size=self.title_size, color="#e8f4fd"),
                x=0.5,
                xanchor="center",
            ),
            paper_bgcolor=self.paper_background,
            plot_bgcolor=self.background,
            margin=self.margin,
            colorway=self.get_color_sequence(),
            legend=dict(
                bgcolor="rgba(17,24,39,0.85)",
                bordercolor="#1f2d45",
                borderwidth=1,
                font=dict(color="#c8d6e5", size=11),
            ),
        )

        # Axes
        axis_style = dict(
            title_font=dict(size=self.axis_title_size, color="#7ba7cc"),
            tickfont=dict(size=self.tick_size, color="#7ba7cc"),
            gridcolor=self.grid_color,
            linecolor="#1f2d45",
            zerolinecolor="#2a3f5c",
            showgrid=True,
            gridwidth=1,
        )
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)

        return fig


# Default theme instance
DEFAULT_THEME = EngineeringTheme()
