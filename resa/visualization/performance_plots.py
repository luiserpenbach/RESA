"""
Performance visualization module using Plotly.

Provides interactive visualizations for:
- Throttle curves
- Performance contours (C*, Isp vs Pc, MR)
- Off-design comparisons
- Phase diagrams
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from resa.visualization.themes import PlotTheme, DEFAULT_THEME

if TYPE_CHECKING:
    from resa.core.results import EngineDesignResult


class ThrottleCurvePlotter:
    """
    Visualizes engine performance across throttle range.

    Shows:
    - Thrust vs throttle setting
    - Chamber pressure vs throttle
    - Mixture ratio drift
    - Wall temperature margins

    Example:
        plotter = ThrottleCurvePlotter()
        fig = plotter.create_figure(throttle_data)
        fig.show()
    """

    def __init__(self, theme: Optional[PlotTheme] = None):
        """Initialize with optional custom theme."""
        self.theme = theme or DEFAULT_THEME

    def create_figure(
        self,
        throttle_data: List[Dict[str, float]],
        design_point: Optional[Dict[str, float]] = None
    ) -> go.Figure:
        """
        Create throttle curve visualization.

        Args:
            throttle_data: List of dicts with keys:
                - 'pct': Throttle percentage
                - 'pc': Chamber pressure [bar]
                - 'mr': Mixture ratio
                - 'thrust': Thrust [N]
                - 'T_wall_max': Max wall temperature [K] (optional)
            design_point: Optional design point to highlight

        Returns:
            Plotly Figure with throttle curves
        """
        # Extract data
        pcts = [d['pct'] for d in throttle_data]
        pcs = [d['pc'] for d in throttle_data]
        mrs = [d['mr'] for d in throttle_data]
        thrusts = [d['thrust'] for d in throttle_data]

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Thrust vs Throttle",
                "Chamber Pressure vs Throttle",
                "Mixture Ratio Drift",
                "Thermal Margin"
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )

        # ========== Plot 1: Thrust ==========
        fig.add_trace(
            go.Scatter(
                x=pcts, y=thrusts,
                mode='lines+markers',
                name='Thrust',
                line=dict(color=self.theme.primary, width=2),
                marker=dict(size=8),
                hovertemplate="Throttle: %{x:.0f}%<br>Thrust: %{y:.0f} N<extra></extra>"
            ),
            row=1, col=1
        )

        # ========== Plot 2: Chamber Pressure ==========
        fig.add_trace(
            go.Scatter(
                x=pcts, y=pcs,
                mode='lines+markers',
                name='Pc',
                line=dict(color=self.theme.secondary, width=2),
                marker=dict(size=8),
                hovertemplate="Throttle: %{x:.0f}%<br>Pc: %{y:.1f} bar<extra></extra>"
            ),
            row=1, col=2
        )

        # ========== Plot 3: Mixture Ratio ==========
        fig.add_trace(
            go.Scatter(
                x=pcts, y=mrs,
                mode='lines+markers',
                name='O/F',
                line=dict(color=self.theme.accent, width=2),
                marker=dict(size=8),
                hovertemplate="Throttle: %{x:.0f}%<br>O/F: %{y:.2f}<extra></extra>"
            ),
            row=2, col=1
        )

        # ========== Plot 4: Wall Temperature ==========
        if 'T_wall_max' in throttle_data[0]:
            T_walls = [d['T_wall_max'] for d in throttle_data]
            fig.add_trace(
                go.Scatter(
                    x=pcts, y=T_walls,
                    mode='lines+markers',
                    name='T_wall_max',
                    line=dict(color=self.theme.danger, width=2),
                    marker=dict(size=8),
                    hovertemplate="Throttle: %{x:.0f}%<br>T_wall: %{y:.0f} K<extra></extra>"
                ),
                row=2, col=2
            )

            # Add temperature limit line (example: 800 K for copper)
            fig.add_hline(
                y=800, row=2, col=2,
                line=dict(color='red', dash='dash', width=1),
                annotation_text="Cu Limit (800 K)",
                annotation_position="top right"
            )

        # ========== Highlight Design Point ==========
        if design_point:
            for row, col, key in [(1, 1, 'thrust'), (1, 2, 'pc'), (2, 1, 'mr')]:
                if key in design_point:
                    fig.add_trace(
                        go.Scatter(
                            x=[design_point.get('pct', 100)],
                            y=[design_point[key]],
                            mode='markers',
                            marker=dict(
                                size=15,
                                symbol='star',
                                color='gold',
                                line=dict(color='black', width=2)
                            ),
                            name='Design Point',
                            showlegend=(row == 1 and col == 1),
                            hoverinfo='skip'
                        ),
                        row=row, col=col
                    )

        # ========== Axis Labels ==========
        fig.update_xaxes(title_text="Throttle [%]", row=1, col=1)
        fig.update_xaxes(title_text="Throttle [%]", row=1, col=2)
        fig.update_xaxes(title_text="Throttle [%]", row=2, col=1)
        fig.update_xaxes(title_text="Throttle [%]", row=2, col=2)

        fig.update_yaxes(title_text="Thrust [N]", row=1, col=1)
        fig.update_yaxes(title_text="Pc [bar]", row=1, col=2)
        fig.update_yaxes(title_text="O/F Ratio [-]", row=2, col=1)
        fig.update_yaxes(title_text="Max Wall Temp [K]", row=2, col=2)

        # ========== Layout ==========
        fig.update_layout(
            height=700,
            title=dict(
                text="Throttle Characterization",
                x=0.5,
                font=dict(size=20)
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )

        self.theme.apply_to_figure(fig)

        return fig

    def to_html(
        self,
        fig: go.Figure,
        include_plotlyjs: str = 'cdn',
        full_html: bool = False
    ) -> str:
        """Export figure to embeddable HTML."""
        return fig.to_html(
            include_plotlyjs=include_plotlyjs,
            full_html=full_html
        )


class PerformanceContourPlotter:
    """
    Creates performance contour maps for engine optimization.

    Visualizes:
    - C* (characteristic velocity) vs Pc and MR
    - Isp contours
    - Operating envelope with constraints

    Example:
        plotter = PerformanceContourPlotter()
        fig = plotter.create_cstar_contour(pc_range, mr_range, cstar_data)
        fig.show()
    """

    def __init__(self, theme: Optional[PlotTheme] = None):
        """Initialize with optional custom theme."""
        self.theme = theme or DEFAULT_THEME

    def create_cstar_contour(
        self,
        pc_values: np.ndarray,
        mr_values: np.ndarray,
        cstar_grid: np.ndarray,
        operating_point: Optional[Dict[str, float]] = None,
        title: str = "C* Performance Map"
    ) -> go.Figure:
        """
        Create C* contour plot.

        Args:
            pc_values: 1D array of chamber pressures [bar]
            mr_values: 1D array of mixture ratios
            cstar_grid: 2D array of C* values [m/s], shape (len(mr), len(pc))
            operating_point: Optional dict with 'pc' and 'mr' keys
            title: Plot title

        Returns:
            Plotly Figure with contour map
        """
        fig = go.Figure()

        # Contour plot
        fig.add_trace(go.Contour(
            x=pc_values,
            y=mr_values,
            z=cstar_grid,
            colorscale='Viridis',
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='white'),
            ),
            colorbar=dict(
                title='C* [m/s]',
                titleside='right',
            ),
            hovertemplate="Pc: %{x:.1f} bar<br>O/F: %{y:.2f}<br>C*: %{z:.0f} m/s<extra></extra>"
        ))

        # Operating point
        if operating_point:
            fig.add_trace(go.Scatter(
                x=[operating_point['pc']],
                y=[operating_point['mr']],
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='cross',
                    color='red',
                    line=dict(color='white', width=2)
                ),
                name='Design Point',
                hovertemplate=f"Design: Pc={operating_point['pc']:.1f} bar, O/F={operating_point['mr']:.2f}<extra></extra>"
            ))

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            xaxis_title="Chamber Pressure [bar]",
            yaxis_title="Mixture Ratio (O/F)",
            width=800,
            height=600
        )

        self.theme.apply_to_figure(fig)

        return fig

    def create_comparison_plot(
        self,
        results: List['EngineDesignResult'],
        labels: List[str],
        metric: str = 'T_wall_hot'
    ) -> go.Figure:
        """
        Compare multiple engine runs on a single plot.

        Args:
            results: List of EngineDesignResult objects
            labels: Labels for each result
            metric: Which metric to compare ('T_wall_hot', 'T_coolant', 'q_flux', etc.)

        Returns:
            Plotly Figure comparing the runs
        """
        fig = go.Figure()

        colors = self.theme.get_color_sequence()

        for i, (result, label) in enumerate(zip(results, labels)):
            x_mm = result.geometry.x_full
            y_data = result.cooling_data[metric]

            # Convert units if needed
            if metric == 'q_flux':
                y_data = y_data / 1e6
                y_label = "Heat Flux [MW/m²]"
            elif metric == 'P_coolant':
                y_data = y_data / 1e5
                y_label = "Pressure [bar]"
            elif 'T_' in metric:
                y_label = "Temperature [K]"
            else:
                y_label = metric

            fig.add_trace(go.Scatter(
                x=x_mm,
                y=y_data,
                mode='lines',
                name=label,
                line=dict(color=colors[i % len(colors)], width=2),
            ))

        fig.update_layout(
            title=dict(
                text=f"Comparison: {metric}",
                x=0.5,
                font=dict(size=18)
            ),
            xaxis_title="Axial Position [mm]",
            yaxis_title=y_label,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            height=500
        )

        self.theme.apply_to_figure(fig)

        return fig

    def to_html(
        self,
        fig: go.Figure,
        include_plotlyjs: str = 'cdn',
        full_html: bool = False
    ) -> str:
        """Export figure to embeddable HTML."""
        return fig.to_html(
            include_plotlyjs=include_plotlyjs,
            full_html=full_html
        )
