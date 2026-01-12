"""
Engine visualization module using Plotly.

Provides interactive visualizations for:
- Engine dashboard (thermal, flow, pressure)
- Cross-section views
- 3D engine geometry
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any, TYPE_CHECKING

from resa.visualization.themes import PlotTheme, EngineeringTheme, DEFAULT_THEME

if TYPE_CHECKING:
    from resa.core.results import EngineDesignResult


class EngineDashboardPlotter:
    """
    Creates the main 4-panel engine analysis dashboard.

    Panels:
    1. Geometry & Thermal State
    2. Gas Dynamics & Heat Flux
    3. Coolant Pressure Evolution
    4. Coolant Flow Properties (Velocity & Density)

    Example:
        plotter = EngineDashboardPlotter()
        fig = plotter.create_figure(engine_result)
        fig.show()  # Interactive display
        html = plotter.to_html(fig)  # For reports
    """

    def __init__(self, theme: Optional[PlotTheme] = None):
        """
        Initialize with optional custom theme.

        Args:
            theme: PlotTheme instance for consistent styling
        """
        self.theme = theme or DEFAULT_THEME

    def create_figure(
        self,
        result: 'EngineDesignResult',
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create the full engine dashboard figure.

        Args:
            result: EngineDesignResult from engine.design() or engine.analyze()
            title: Optional custom title

        Returns:
            Plotly Figure with 4 subplots
        """
        # Create 4-row subplot with secondary y-axes
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                "1. Chamber Geometry & Thermal State",
                "2. Gas Dynamics & Heat Flux Profile",
                "3. Coolant Pressure Drop",
                "4. Coolant Hydraulics (Velocity & Density)"
            ),
            specs=[
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"secondary_y": False}],
                [{"secondary_y": True}]
            ]
        )

        # Extract data arrays
        x_mm = result.geometry.x_full
        y_mm = result.geometry.y_full
        cooling = result.cooling_data
        mach = result.mach_numbers

        # ========== Panel 1: Geometry & Temperature ==========
        # Wall contour (primary y)
        fig.add_trace(
            go.Scatter(
                x=x_mm, y=y_mm,
                name="Chamber Wall",
                mode='lines',
                line=dict(color='black', width=3),
                fill='tozeroy',
                fillcolor='rgba(184, 115, 51, 0.15)',
                hovertemplate="X: %{x:.1f} mm<br>R: %{y:.2f} mm<extra></extra>"
            ),
            row=1, col=1, secondary_y=False
        )

        # Hot wall temperature (secondary y)
        fig.add_trace(
            go.Scatter(
                x=x_mm, y=cooling['T_wall_hot'],
                name="Hot Wall Temp",
                mode='lines',
                line=dict(color=self.theme.danger, width=2),
                hovertemplate="X: %{x:.1f} mm<br>T_wall: %{y:.0f} K<extra></extra>"
            ),
            row=1, col=1, secondary_y=True
        )

        # Coolant temperature (secondary y)
        fig.add_trace(
            go.Scatter(
                x=x_mm, y=cooling['T_coolant'],
                name="Coolant Temp",
                mode='lines',
                line=dict(color=self.theme.primary, width=2, dash='dash'),
                hovertemplate="X: %{x:.1f} mm<br>T_coolant: %{y:.0f} K<extra></extra>"
            ),
            row=1, col=1, secondary_y=True
        )

        # ========== Panel 2: Mach & Heat Flux ==========
        # Mach number (primary y)
        fig.add_trace(
            go.Scatter(
                x=x_mm, y=mach,
                name="Mach Number",
                mode='lines',
                line=dict(color=self.theme.accent, width=2),
                hovertemplate="X: %{x:.1f} mm<br>Mach: %{y:.2f}<extra></extra>"
            ),
            row=2, col=1, secondary_y=False
        )

        # Heat flux (secondary y)
        q_flux_mw = cooling['q_flux'] / 1e6
        fig.add_trace(
            go.Scatter(
                x=x_mm, y=q_flux_mw,
                name="Heat Flux",
                mode='lines',
                line=dict(color='#9b59b6', width=2),
                hovertemplate="X: %{x:.1f} mm<br>q: %{y:.2f} MW/m²<extra></extra>"
            ),
            row=2, col=1, secondary_y=True
        )

        # ========== Panel 3: Coolant Pressure ==========
        p_bar = cooling['P_coolant'] / 1e5
        fig.add_trace(
            go.Scatter(
                x=x_mm, y=p_bar,
                name="Coolant Pressure",
                mode='lines',
                line=dict(color=self.theme.info, width=2.5),
                fill='tozeroy',
                fillcolor='rgba(23, 162, 184, 0.15)',
                hovertemplate="X: %{x:.1f} mm<br>P: %{y:.1f} bar<extra></extra>"
            ),
            row=3, col=1
        )

        # ========== Panel 4: Velocity & Density ==========
        # Velocity (primary y)
        fig.add_trace(
            go.Scatter(
                x=x_mm, y=cooling['velocity'],
                name="Velocity",
                mode='lines',
                line=dict(color='black', width=2),
                hovertemplate="X: %{x:.1f} mm<br>V: %{y:.1f} m/s<extra></extra>"
            ),
            row=4, col=1, secondary_y=False
        )

        # ========== Axis Labels ==========
        # Row 1
        fig.update_yaxes(title_text="Radius [mm]", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Temperature [K]", row=1, col=1, secondary_y=True,
                         title_font=dict(color=self.theme.danger))

        # Row 2
        fig.update_yaxes(title_text="Mach Number [-]", row=2, col=1, secondary_y=False,
                         title_font=dict(color=self.theme.accent))
        fig.update_yaxes(title_text="Heat Flux [MW/m²]", row=2, col=1, secondary_y=True,
                         title_font=dict(color='#9b59b6'))

        # Row 3
        fig.update_yaxes(title_text="Pressure [bar]", row=3, col=1)

        # Row 4
        fig.update_yaxes(title_text="Velocity [m/s]", row=4, col=1, secondary_y=False)

        # X-axis label (only on bottom)
        fig.update_xaxes(title_text="Axial Position [mm]", row=4, col=1)

        # ========== Layout ==========
        engine_name = getattr(result, 'engine_name', 'Engine')
        if hasattr(result, 'config'):
            engine_name = result.config.engine_name

        fig.update_layout(
            height=1200,
            title=dict(
                text=title or f"Engine Analysis: {engine_name}",
                x=0.5,
                font=dict(size=22)
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255,255,255,0.8)',
            ),
            hovermode='x unified',
        )

        # Apply theme
        self.theme.apply_to_figure(fig)

        return fig

    def to_html(
        self,
        fig: go.Figure,
        include_plotlyjs: str = 'cdn',
        full_html: bool = False
    ) -> str:
        """
        Export figure to embeddable HTML.

        Args:
            fig: Plotly Figure
            include_plotlyjs: 'cdn', True, False, or path to local plotly.js
            full_html: If True, includes <html> tags

        Returns:
            HTML string
        """
        return fig.to_html(
            include_plotlyjs=include_plotlyjs,
            full_html=full_html
        )

    def show(self, fig: go.Figure) -> None:
        """Display figure interactively."""
        fig.show()


class CrossSectionPlotter:
    """
    Visualizes cooling channel cross-sections in polar/radial view.

    Creates an accurate representation of:
    - Inner liner (hot wall)
    - Cooling channels (coolant domain)
    - Ribs (land between channels)
    - Outer closeout/jacket

    Example:
        plotter = CrossSectionPlotter()
        fig = plotter.create_figure(channel_geometry, station_idx=50)
        fig.show()
    """

    def __init__(self, theme: Optional[PlotTheme] = None):
        """Initialize with optional custom theme."""
        self.theme = theme or DEFAULT_THEME

    def create_figure(
        self,
        channel_geometry: Any,
        station_idx: int,
        sector_angle: float = 90.0,
        closeout_thickness: float = 0.001
    ) -> go.Figure:
        """
        Create radial cross-section view at a given axial station.

        Args:
            channel_geometry: CoolingChannelGeometry object
            station_idx: Index along the axial direction
            sector_angle: Degrees of the sector to display (360 for full)
            closeout_thickness: Outer jacket thickness in meters

        Returns:
            Plotly Figure with the cross-section
        """
        geo = channel_geometry

        # Extract dimensions at this station (convert to mm)
        R_inner = geo.y[station_idx] * 1000
        t_wall = geo.wall_thickness[station_idx] * 1000
        w_ch = geo.channel_width[station_idx] * 1000
        h_ch = geo.channel_height[station_idx] * 1000
        w_rib = geo.rib_width[station_idx] * 1000
        n_channels = geo.num_channels
        t_closeout = closeout_thickness * 1000

        # Calculate radii
        R_channel_base = R_inner + t_wall
        R_channel_top = R_channel_base + h_ch
        R_outer = R_channel_top + t_closeout

        # Angular calculations
        theta_rad = np.radians(sector_angle)
        theta_points = 100

        fig = go.Figure()

        # ========== Layer 1: Inner Liner ==========
        theta = np.linspace(0, theta_rad, theta_points)
        # Create closed polygon for liner
        r_inner_x = np.concatenate([
            R_inner * np.cos(theta),
            R_channel_base * np.cos(theta[::-1])
        ])
        r_inner_y = np.concatenate([
            R_inner * np.sin(theta),
            R_channel_base * np.sin(theta[::-1])
        ])

        fig.add_trace(go.Scatter(
            x=r_inner_x, y=r_inner_y,
            fill='toself',
            fillcolor=self.theme.copper,
            line=dict(color='black', width=1),
            name='Liner (Cu)',
            hoverinfo='name'
        ))

        # ========== Layer 2: Channels and Ribs ==========
        theta_pitch = 2 * np.pi / n_channels
        channel_fraction = w_ch / (w_ch + w_rib)
        theta_ch = theta_pitch * channel_fraction

        channels_to_draw = int(np.ceil(theta_rad / theta_pitch)) + 1

        for i in range(channels_to_draw):
            start_angle = i * theta_pitch

            # Skip if beyond sector
            if start_angle > theta_rad:
                break

            # Channel end angle (clamp to sector)
            ch_end = min(start_angle + theta_ch, theta_rad)
            if ch_end <= start_angle:
                continue

            # Draw channel (coolant)
            ch_theta = np.linspace(start_angle, ch_end, 20)
            ch_x = np.concatenate([
                R_channel_base * np.cos(ch_theta),
                R_channel_top * np.cos(ch_theta[::-1])
            ])
            ch_y = np.concatenate([
                R_channel_base * np.sin(ch_theta),
                R_channel_top * np.sin(ch_theta[::-1])
            ])

            fig.add_trace(go.Scatter(
                x=ch_x, y=ch_y,
                fill='toself',
                fillcolor=self.theme.coolant,
                line=dict(color='black', width=0.5),
                name='Coolant' if i == 0 else None,
                showlegend=(i == 0),
                hoverinfo='name' if i == 0 else 'skip'
            ))

            # Rib (if within sector)
            rib_start = start_angle + theta_ch
            rib_end = min(start_angle + theta_pitch, theta_rad)
            if rib_end > rib_start:
                rib_theta = np.linspace(rib_start, rib_end, 20)
                rib_x = np.concatenate([
                    R_channel_base * np.cos(rib_theta),
                    R_channel_top * np.cos(rib_theta[::-1])
                ])
                rib_y = np.concatenate([
                    R_channel_base * np.sin(rib_theta),
                    R_channel_top * np.sin(rib_theta[::-1])
                ])

                fig.add_trace(go.Scatter(
                    x=rib_x, y=rib_y,
                    fill='toself',
                    fillcolor=self.theme.copper,
                    line=dict(color='black', width=0.5),
                    showlegend=False,
                    hoverinfo='skip'
                ))

        # ========== Layer 3: Closeout Jacket ==========
        jacket_x = np.concatenate([
            R_channel_top * np.cos(theta),
            R_outer * np.cos(theta[::-1])
        ])
        jacket_y = np.concatenate([
            R_channel_top * np.sin(theta),
            R_outer * np.sin(theta[::-1])
        ])

        fig.add_trace(go.Scatter(
            x=jacket_x, y=jacket_y,
            fill='toself',
            fillcolor=self.theme.steel,
            line=dict(color='black', width=1),
            name='Closeout',
            hoverinfo='name'
        ))

        # ========== Annotations ==========
        x_pos = geo.x[station_idx] * 1000
        fig.add_annotation(
            x=0, y=0,
            text=f"N = {n_channels}<br>X = {x_pos:.1f} mm",
            showarrow=False,
            font=dict(size=12, color='black'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1,
            borderpad=4,
        )

        # Dimension annotations
        fig.add_annotation(
            x=R_outer * 1.15, y=0,
            text=f"ID: {2*R_inner:.1f} mm<br>OD: {2*R_outer:.1f} mm",
            showarrow=False,
            font=dict(size=10),
            align='left'
        )

        # ========== Layout ==========
        limit = R_outer * 1.3

        fig.update_layout(
            title=dict(
                text=f"Radial Cross-Section at Station {station_idx}",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-limit * 0.2, limit]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-limit * 0.2, limit]
            ),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.9)',
            ),
            width=700,
            height=700,
            margin=dict(l=20, r=20, t=60, b=20)
        )

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


class ContourPlotter:
    """
    Visualizes engine contour with temperature overlay.

    Creates a 2D representation of the engine geometry with:
    - Wall contour profile
    - Temperature color mapping on the wall
    - Annotated key stations (chamber, throat, exit)
    - Cooling channel indicators

    Example:
        plotter = ContourPlotter()
        fig = plotter.create_figure(engine_result)
        fig.show()
    """

    def __init__(self, theme: Optional[PlotTheme] = None):
        """Initialize with optional custom theme."""
        self.theme = theme or DEFAULT_THEME

    def create_figure(
        self,
        result: 'EngineDesignResult',
        show_channels: bool = True,
        show_annotations: bool = True,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create engine contour visualization with temperature overlay.

        Args:
            result: EngineDesignResult from engine.design() or engine.analyze()
            show_channels: If True, shows cooling channel indicators
            show_annotations: If True, adds station annotations
            title: Optional custom title

        Returns:
            Plotly Figure with engine contour and temperature overlay
        """
        fig = go.Figure()

        # Extract data
        x_mm = result.geometry.x_full
        y_mm = result.geometry.y_full
        cooling = result.cooling_data
        T_wall = cooling['T_wall_hot']

        # Normalize temperature for color mapping
        T_min, T_max = T_wall.min(), T_wall.max()
        T_norm = (T_wall - T_min) / (T_max - T_min + 1e-6)

        # Create colorscale from theme
        colorscale = self.theme.temperature_colorscale

        # ========== Upper wall contour with temperature ==========
        # Create filled contour using scatter with colorscale
        for i in range(len(x_mm) - 1):
            # Get color from normalized temperature
            color_idx = T_norm[i]
            # Interpolate color from colorscale
            color = self._interpolate_colorscale(colorscale, color_idx)

            fig.add_trace(go.Scatter(
                x=[x_mm[i], x_mm[i + 1], x_mm[i + 1], x_mm[i], x_mm[i]],
                y=[y_mm[i], y_mm[i + 1], 0, 0, y_mm[i]],
                fill='toself',
                fillcolor=color,
                line=dict(width=0),
                mode='lines',
                showlegend=False,
                hoverinfo='skip'
            ))

        # ========== Wall outline ==========
        # Upper wall
        fig.add_trace(go.Scatter(
            x=x_mm, y=y_mm,
            mode='lines',
            line=dict(color='black', width=2),
            name='Wall Contour',
            hovertemplate="X: %{x:.1f} mm<br>R: %{y:.2f} mm<extra></extra>"
        ))

        # Lower wall (mirror)
        fig.add_trace(go.Scatter(
            x=x_mm, y=-np.array(y_mm),
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Mirror the temperature fill
        for i in range(len(x_mm) - 1):
            color_idx = T_norm[i]
            color = self._interpolate_colorscale(colorscale, color_idx)

            fig.add_trace(go.Scatter(
                x=[x_mm[i], x_mm[i + 1], x_mm[i + 1], x_mm[i], x_mm[i]],
                y=[-y_mm[i], -y_mm[i + 1], 0, 0, -y_mm[i]],
                fill='toself',
                fillcolor=color,
                line=dict(width=0),
                mode='lines',
                showlegend=False,
                hoverinfo='skip'
            ))

        # ========== Cooling channels (simplified indicator) ==========
        if show_channels:
            channel_offset = 2  # mm offset for channel indication
            fig.add_trace(go.Scatter(
                x=x_mm, y=np.array(y_mm) + channel_offset,
                mode='lines',
                line=dict(color=self.theme.coolant, width=3, dash='dot'),
                name='Cooling Circuit',
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=x_mm, y=-np.array(y_mm) - channel_offset,
                mode='lines',
                line=dict(color=self.theme.coolant, width=3, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # ========== Annotations ==========
        if show_annotations:
            # Find throat position (minimum radius)
            throat_idx = np.argmin(y_mm)
            throat_x = x_mm[throat_idx]
            throat_r = y_mm[throat_idx]

            # Chamber end (first index typically)
            chamber_x = x_mm[0]
            chamber_r = y_mm[0]

            # Exit (last index)
            exit_x = x_mm[-1]
            exit_r = y_mm[-1]

            annotations = [
                dict(
                    x=chamber_x, y=chamber_r + 5,
                    text=f"Chamber<br>D={2*chamber_r:.1f}mm",
                    showarrow=True, arrowhead=2, ax=0, ay=-30,
                    font=dict(size=10)
                ),
                dict(
                    x=throat_x, y=throat_r + 5,
                    text=f"Throat<br>D*={2*throat_r:.1f}mm",
                    showarrow=True, arrowhead=2, ax=0, ay=-30,
                    font=dict(size=10, color=self.theme.danger)
                ),
                dict(
                    x=exit_x, y=exit_r + 5,
                    text=f"Exit<br>D={2*exit_r:.1f}mm",
                    showarrow=True, arrowhead=2, ax=0, ay=-30,
                    font=dict(size=10)
                )
            ]

            for ann in annotations:
                fig.add_annotation(**ann)

        # ========== Temperature colorbar ==========
        # Add invisible trace for colorbar
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale=colorscale,
                cmin=T_min,
                cmax=T_max,
                colorbar=dict(
                    title='Wall Temp [K]',
                    titleside='right',
                    thickness=15,
                    len=0.7,
                    y=0.5
                ),
                showscale=True
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

        # ========== Layout ==========
        engine_name = getattr(result, 'engine_name', 'Engine')
        if hasattr(result, 'config'):
            engine_name = result.config.engine_name

        max_r = max(y_mm)
        fig.update_layout(
            title=dict(
                text=title or f"Engine Contour: {engine_name}",
                x=0.5,
                font=dict(size=18)
            ),
            xaxis=dict(
                title="Axial Position [mm]",
                zeroline=False,
                showgrid=True,
                gridcolor=self.theme.grid_color
            ),
            yaxis=dict(
                title="Radial Position [mm]",
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1,
                showgrid=True,
                gridcolor=self.theme.grid_color,
                scaleanchor="x",
                scaleratio=1,
                range=[-max_r * 1.5, max_r * 1.5]
            ),
            showlegend=True,
            legend=dict(
                x=0.02, y=0.98,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            height=600,
            width=1000,
            margin=dict(l=60, r=100, t=80, b=60)
        )

        self.theme.apply_to_figure(fig)

        return fig

    def _interpolate_colorscale(self, colorscale: list, value: float) -> str:
        """
        Interpolate color from colorscale at given normalized value.

        Args:
            colorscale: List of [position, color] pairs
            value: Normalized value between 0 and 1

        Returns:
            Interpolated color string
        """
        value = max(0, min(1, value))

        # Find bracketing colors
        lower_idx = 0
        for i, (pos, _) in enumerate(colorscale):
            if pos <= value:
                lower_idx = i

        if lower_idx >= len(colorscale) - 1:
            return colorscale[-1][1]

        lower_pos, lower_color = colorscale[lower_idx]
        upper_pos, upper_color = colorscale[lower_idx + 1]

        # Interpolate
        if upper_pos == lower_pos:
            t = 0
        else:
            t = (value - lower_pos) / (upper_pos - lower_pos)

        # Parse colors and interpolate RGB
        lower_rgb = self._parse_color(lower_color)
        upper_rgb = self._parse_color(upper_color)

        r = int(lower_rgb[0] + t * (upper_rgb[0] - lower_rgb[0]))
        g = int(lower_rgb[1] + t * (upper_rgb[1] - lower_rgb[1]))
        b = int(lower_rgb[2] + t * (upper_rgb[2] - lower_rgb[2]))

        return f'rgb({r},{g},{b})'

    def _parse_color(self, color: str) -> tuple:
        """Parse hex color to RGB tuple."""
        color = color.lstrip('#')
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

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
