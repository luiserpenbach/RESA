"""
Injector visualization module using Plotly.

Provides interactive visualizations for:
- Swirl injector cross-sections
- Spray pattern visualization
- Injector element details
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass

from resa.visualization.themes import PlotTheme, EngineeringTheme, DEFAULT_THEME

if TYPE_CHECKING:
    pass


@dataclass
class InjectorGeometry:
    """
    Injector geometry data container.

    Attributes:
        orifice_radius: Orifice radius [m]
        swirl_chamber_radius: Swirl chamber radius [m]
        port_radius: Tangential port radius [m]
        orifice_length: Orifice length [m]
        swirl_chamber_length: Swirl chamber length [m]
        num_ports: Number of tangential ports
        oxidizer_port_radius: Oxidizer port radius [m] (for LCSC)
    """
    orifice_radius: float
    swirl_chamber_radius: float
    port_radius: float
    orifice_length: float = None
    swirl_chamber_length: float = None
    num_ports: int = 3
    oxidizer_port_radius: float = None


class InjectorCrossSectionPlotter:
    """
    Creates cross-section visualization of swirl injector elements.

    Supports both:
    - LCSC (Liquid-Centered Swirl Coaxial) injectors
    - GCSC (Gas-Centered Swirl Coaxial) injectors

    Example:
        plotter = InjectorCrossSectionPlotter()
        fig = plotter.create_figure(geometry)
        fig.show()
    """

    def __init__(self, theme: Optional[PlotTheme] = None):
        """Initialize with optional custom theme."""
        self.theme = theme or DEFAULT_THEME

    def create_figure(
        self,
        geometry: InjectorGeometry,
        injector_type: str = "LCSC",
        title: Optional[str] = None,
        show_dimensions: bool = True,
        show_flow_arrows: bool = True
    ) -> go.Figure:
        """
        Create injector cross-section visualization.

        Args:
            geometry: InjectorGeometry dataclass with dimensions
            injector_type: "LCSC" or "GCSC"
            title: Optional custom title
            show_dimensions: If True, shows dimension annotations
            show_flow_arrows: If True, shows flow direction arrows

        Returns:
            Plotly Figure with injector cross-section
        """
        fig = go.Figure()

        # Convert to mm for display
        r_o = geometry.orifice_radius * 1000
        r_sc = geometry.swirl_chamber_radius * 1000
        r_p = geometry.port_radius * 1000
        n_p = geometry.num_ports

        # Lengths (estimate if not provided)
        l_o = (geometry.orifice_length or geometry.orifice_radius * 2) * 1000
        l_sc = (geometry.swirl_chamber_length or geometry.swirl_chamber_radius * 3) * 1000

        # ========== Swirl Chamber ==========
        fig.add_shape(
            type="rect",
            x0=0, y0=-r_sc, x1=l_sc, y1=r_sc,
            line=dict(color="black", width=2),
            fillcolor=self.theme.coolant,
            opacity=0.3,
            layer="below"
        )

        # Chamber outline trace for legend
        fig.add_trace(go.Scatter(
            x=[0, l_sc, l_sc, 0, 0],
            y=[r_sc, r_sc, -r_sc, -r_sc, r_sc],
            mode='lines',
            line=dict(color='black', width=2),
            fill='toself',
            fillcolor=self.theme.coolant,
            opacity=0.3,
            name='Swirl Chamber',
            hovertemplate=f"Swirl Chamber<br>D = {2*r_sc:.2f} mm<extra></extra>"
        ))

        # ========== Orifice ==========
        fig.add_shape(
            type="rect",
            x0=l_sc, y0=-r_o, x1=l_sc + l_o, y1=r_o,
            line=dict(color="black", width=2),
            fillcolor=self.theme.coolant,
            opacity=0.5,
            layer="below"
        )

        fig.add_trace(go.Scatter(
            x=[l_sc, l_sc + l_o, l_sc + l_o, l_sc, l_sc],
            y=[r_o, r_o, -r_o, -r_o, r_o],
            mode='lines',
            line=dict(color='black', width=2),
            fill='toself',
            fillcolor=self.theme.coolant,
            opacity=0.5,
            name='Orifice',
            hovertemplate=f"Orifice<br>D = {2*r_o:.2f} mm<extra></extra>"
        ))

        # ========== Oxidizer Port (for LCSC) ==========
        if injector_type == "LCSC" and geometry.oxidizer_port_radius:
            r_ox = geometry.oxidizer_port_radius * 1000
            l_ox = 5  # Extension length in mm

            fig.add_shape(
                type="rect",
                x0=l_sc + l_o, y0=-r_ox, x1=l_sc + l_o + l_ox, y1=r_ox,
                line=dict(color="black", width=2),
                fillcolor=self.theme.oxidizer,
                opacity=0.3,
                layer="below"
            )

            fig.add_trace(go.Scatter(
                x=[l_sc + l_o, l_sc + l_o + l_ox, l_sc + l_o + l_ox, l_sc + l_o, l_sc + l_o],
                y=[r_ox, r_ox, -r_ox, -r_ox, r_ox],
                mode='lines',
                line=dict(color='black', width=2),
                fill='toself',
                fillcolor=self.theme.oxidizer,
                opacity=0.3,
                name='Oxidizer Port',
                hovertemplate=f"Oxidizer Port<br>D = {2*r_ox:.2f} mm<extra></extra>"
            ))

        # ========== Tangential Ports ==========
        # Draw ports around swirl chamber (simplified 2D view - show as circles on side)
        port_positions = np.linspace(l_sc * 0.2, l_sc * 0.8, min(n_p, 3))

        for i, x_pos in enumerate(port_positions):
            # Upper port
            fig.add_shape(
                type="circle",
                x0=x_pos - r_p, y0=r_sc - r_p,
                x1=x_pos + r_p, y1=r_sc + r_p,
                line=dict(color="black", width=1.5),
                fillcolor=self.theme.fuel,
                opacity=0.5,
            )

            # Lower port (symmetric)
            fig.add_shape(
                type="circle",
                x0=x_pos - r_p, y0=-r_sc - r_p,
                x1=x_pos + r_p, y1=-r_sc + r_p,
                line=dict(color="black", width=1.5),
                fillcolor=self.theme.fuel,
                opacity=0.5,
            )

        # Add port legend entry
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=self.theme.fuel, symbol='circle'),
            name=f'Tangential Ports (n={n_p})',
            showlegend=True
        ))

        # ========== Flow Arrows ==========
        if show_flow_arrows:
            # Fuel swirl flow
            arrow_y = r_sc * 0.6
            fig.add_annotation(
                x=l_sc * 0.5, y=arrow_y,
                ax=l_sc * 0.2, ay=arrow_y,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=self.theme.fuel
            )

            # Axial flow through orifice
            fig.add_annotation(
                x=l_sc + l_o * 0.9, y=0,
                ax=l_sc + l_o * 0.3, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor='black'
            )

            # Tangential port flow indicator
            fig.add_annotation(
                x=port_positions[0], y=r_sc,
                ax=port_positions[0], ay=r_sc + r_p * 2,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=1.5,
                arrowcolor=self.theme.fuel
            )

        # ========== Dimension Annotations ==========
        if show_dimensions:
            annotations = [
                dict(
                    x=l_sc / 2, y=r_sc + 2,
                    text=f"Swirl Chamber: D={2*r_sc:.2f} mm",
                    showarrow=False,
                    font=dict(size=10, color='black'),
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                dict(
                    x=l_sc + l_o / 2, y=r_o + 1.5,
                    text=f"Orifice: D={2*r_o:.2f} mm",
                    showarrow=False,
                    font=dict(size=10, color='black'),
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                dict(
                    x=-2, y=0,
                    text=f"L_sc = {l_sc:.1f} mm",
                    showarrow=False,
                    font=dict(size=9),
                    textangle=-90
                ),
            ]

            for ann in annotations:
                fig.add_annotation(**ann)

        # ========== Centerline ==========
        fig.add_trace(go.Scatter(
            x=[0, l_sc + l_o + 10],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', dash='dashdot', width=1),
            name='Centerline',
            showlegend=False,
            hoverinfo='skip'
        ))

        # ========== Air Core (vortex visualization) ==========
        # Show air core in orifice
        core_fraction = 0.3  # Typical air core fraction
        r_core = r_o * core_fraction

        theta = np.linspace(0, 2 * np.pi, 50)
        x_spiral = []
        y_spiral = []
        for t in np.linspace(0, 3 * np.pi, 100):
            r = r_o * (1 - 0.3 * t / (3 * np.pi))
            x_spiral.append(l_sc + l_o * 0.5 + 0.5 * np.sin(t))
            y_spiral.append(r * np.cos(t))

        fig.add_trace(go.Scatter(
            x=x_spiral,
            y=y_spiral,
            mode='lines',
            line=dict(color='rgba(0,0,0,0.3)', width=1, dash='dot'),
            name='Vortex Flow',
            showlegend=False,
            hoverinfo='skip'
        ))

        # ========== Layout ==========
        max_y = r_sc * 1.5
        max_x = l_sc + l_o + 15

        fig.update_layout(
            title=dict(
                text=title or f"{injector_type} Injector Cross-Section",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis=dict(
                title="Axial Position [mm]",
                scaleanchor="y",
                scaleratio=1,
                showgrid=True,
                gridcolor=self.theme.grid_color,
                range=[-5, max_x]
            ),
            yaxis=dict(
                title="Radial Position [mm]",
                showgrid=True,
                gridcolor=self.theme.grid_color,
                range=[-max_y, max_y]
            ),
            showlegend=True,
            legend=dict(
                x=1.02, y=0.98,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='black',
                borderwidth=1
            ),
            height=500,
            width=800,
            margin=dict(l=60, r=150, t=80, b=60)
        )

        self.theme.apply_to_figure(fig)

        return fig

    def create_comparison_figure(
        self,
        geometries: List[InjectorGeometry],
        labels: List[str],
        title: str = "Injector Design Comparison"
    ) -> go.Figure:
        """
        Create side-by-side comparison of injector designs.

        Args:
            geometries: List of InjectorGeometry objects
            labels: Labels for each design
            title: Plot title

        Returns:
            Plotly Figure with comparison subplots
        """
        n = len(geometries)
        fig = make_subplots(
            rows=1, cols=n,
            subplot_titles=labels,
            horizontal_spacing=0.1
        )

        colors = self.theme.get_color_sequence()

        for i, (geom, label) in enumerate(zip(geometries, labels), 1):
            # Convert to mm
            r_o = geom.orifice_radius * 1000
            r_sc = geom.swirl_chamber_radius * 1000
            l_o = (geom.orifice_length or geom.orifice_radius * 2) * 1000
            l_sc = (geom.swirl_chamber_length or geom.swirl_chamber_radius * 3) * 1000

            # Swirl chamber
            fig.add_trace(
                go.Scatter(
                    x=[0, l_sc, l_sc, 0, 0],
                    y=[r_sc, r_sc, -r_sc, -r_sc, r_sc],
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=2),
                    fill='toself',
                    fillcolor=f'rgba{tuple(list(self._hex_to_rgb(colors[i % len(colors)])) + [0.2])}',
                    name=f'{label} Chamber',
                    showlegend=(i == 1)
                ),
                row=1, col=i
            )

            # Orifice
            fig.add_trace(
                go.Scatter(
                    x=[l_sc, l_sc + l_o, l_sc + l_o, l_sc, l_sc],
                    y=[r_o, r_o, -r_o, -r_o, r_o],
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=2),
                    fill='toself',
                    fillcolor=f'rgba{tuple(list(self._hex_to_rgb(colors[i % len(colors)])) + [0.4])}',
                    name=f'{label} Orifice',
                    showlegend=False
                ),
                row=1, col=i
            )

            fig.update_xaxes(title_text="X [mm]", row=1, col=i)
            fig.update_yaxes(title_text="R [mm]" if i == 1 else "", row=1, col=i)

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            height=400,
            width=400 * n,
            showlegend=True
        )

        self.theme.apply_to_figure(fig)

        return fig

    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

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


class SprayPatternPlotter:
    """
    Visualizes spray patterns from swirl injectors.

    Creates visualizations showing:
    - Spray cone angle
    - Droplet distribution
    - Film thickness variation
    - Spray penetration

    Example:
        plotter = SprayPatternPlotter()
        fig = plotter.create_figure(spray_angle=60, cone_length=50)
        fig.show()
    """

    def __init__(self, theme: Optional[PlotTheme] = None):
        """Initialize with optional custom theme."""
        self.theme = theme or DEFAULT_THEME

    def create_figure(
        self,
        spray_half_angle: float,
        orifice_diameter: float,
        cone_length: float = 50.0,
        film_thickness: Optional[float] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create spray pattern visualization.

        Args:
            spray_half_angle: Half cone angle in degrees
            orifice_diameter: Orifice diameter in mm
            cone_length: Spray cone length to visualize in mm
            film_thickness: Liquid film thickness at orifice [mm] (optional)
            title: Optional custom title

        Returns:
            Plotly Figure with spray pattern
        """
        fig = go.Figure()

        # Convert angle to radians
        angle_rad = np.radians(spray_half_angle)

        # Orifice radius
        r_o = orifice_diameter / 2

        # ========== Orifice Exit Plane ==========
        fig.add_trace(go.Scatter(
            x=[0, 0],
            y=[-r_o * 2, r_o * 2],
            mode='lines',
            line=dict(color='black', width=3),
            name='Orifice Exit',
            hoverinfo='skip'
        ))

        # ========== Spray Cone ==========
        # Outer cone boundary
        x_cone = np.array([0, cone_length])
        y_upper = np.array([r_o, r_o + cone_length * np.tan(angle_rad)])
        y_lower = np.array([-r_o, -r_o - cone_length * np.tan(angle_rad)])

        # Fill the spray cone
        x_fill = np.concatenate([[0], x_cone, x_cone[::-1], [0]])
        y_fill = np.concatenate([[r_o], y_upper, y_lower[::-1], [-r_o]])

        fig.add_trace(go.Scatter(
            x=x_fill,
            y=y_fill,
            fill='toself',
            fillcolor=self.theme.spray_cone,
            line=dict(color=self.theme.primary, width=2),
            name='Spray Cone',
            hovertemplate=f"Spray Half Angle: {spray_half_angle:.1f} deg<extra></extra>"
        ))

        # ========== Air Core ==========
        if film_thickness:
            # Air core (hollow cone interior)
            r_core = r_o - film_thickness
            if r_core > 0:
                # Inner cone boundary
                y_inner_upper = np.array([r_core, r_core + cone_length * np.tan(angle_rad) * 0.8])
                y_inner_lower = np.array([-r_core, -r_core - cone_length * np.tan(angle_rad) * 0.8])

                x_air = np.concatenate([[0], x_cone, x_cone[::-1], [0]])
                y_air = np.concatenate([[r_core], y_inner_upper, y_inner_lower[::-1], [-r_core]])

                fig.add_trace(go.Scatter(
                    x=x_air,
                    y=y_air,
                    fill='toself',
                    fillcolor='rgba(255,255,255,0.8)',
                    line=dict(color='gray', width=1, dash='dash'),
                    name='Air Core',
                    hoverinfo='skip'
                ))

        # ========== Droplet Indicators ==========
        # Add scattered points to represent droplets
        np.random.seed(42)
        n_droplets = 100

        # Generate droplets within cone
        for i in range(n_droplets):
            x_drop = np.random.uniform(5, cone_length * 0.9)
            max_r = r_o + x_drop * np.tan(angle_rad)
            min_r = 0 if not film_thickness else max(0, (r_o - film_thickness) + x_drop * np.tan(angle_rad) * 0.8)

            # Random radial position (more likely near edges for hollow cone)
            if np.random.random() > 0.3:  # 70% near outer edge
                r_drop = np.random.uniform(max_r * 0.7, max_r)
            else:
                r_drop = np.random.uniform(min_r, max_r * 0.7)

            # Random sign for upper/lower
            if np.random.random() > 0.5:
                r_drop = -r_drop

            size = np.random.uniform(2, 6)
            opacity = np.random.uniform(0.3, 0.7)

            fig.add_trace(go.Scatter(
                x=[x_drop],
                y=[r_drop],
                mode='markers',
                marker=dict(
                    size=size,
                    color=self.theme.coolant,
                    opacity=opacity
                ),
                showlegend=False,
                hoverinfo='skip'
            ))

        # ========== Angle Annotation Arc ==========
        arc_radius = cone_length * 0.3
        arc_angles = np.linspace(0, angle_rad, 20)
        arc_x = arc_radius * np.cos(arc_angles)
        arc_y = arc_radius * np.sin(arc_angles)

        fig.add_trace(go.Scatter(
            x=arc_x,
            y=arc_y,
            mode='lines',
            line=dict(color='black', width=1.5),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Angle label
        fig.add_annotation(
            x=arc_radius * 1.1 * np.cos(angle_rad / 2),
            y=arc_radius * 1.1 * np.sin(angle_rad / 2),
            text=f"{spray_half_angle:.1f}",
            showarrow=False,
            font=dict(size=12, color='black')
        )

        # ========== Centerline ==========
        fig.add_trace(go.Scatter(
            x=[0, cone_length * 1.1],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', dash='dashdot', width=1),
            name='Centerline',
            showlegend=False,
            hoverinfo='skip'
        ))

        # ========== Layout ==========
        max_r = r_o + cone_length * np.tan(angle_rad)

        fig.update_layout(
            title=dict(
                text=title or f"Spray Pattern (2alpha = {2*spray_half_angle:.1f} deg)",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis=dict(
                title="Axial Distance from Orifice [mm]",
                showgrid=True,
                gridcolor=self.theme.grid_color,
                range=[-5, cone_length * 1.15],
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                title="Radial Position [mm]",
                showgrid=True,
                gridcolor=self.theme.grid_color,
                range=[-max_r * 1.2, max_r * 1.2]
            ),
            showlegend=True,
            legend=dict(
                x=1.02, y=0.98,
                bgcolor='rgba(255,255,255,0.9)'
            ),
            height=600,
            width=900,
            margin=dict(l=60, r=150, t=80, b=60)
        )

        self.theme.apply_to_figure(fig)

        return fig

    def create_spray_comparison(
        self,
        configurations: List[Dict[str, float]],
        labels: List[str],
        title: str = "Spray Pattern Comparison"
    ) -> go.Figure:
        """
        Compare spray patterns from different injector configurations.

        Args:
            configurations: List of dicts with 'spray_half_angle' and 'orifice_diameter'
            labels: Labels for each configuration
            title: Plot title

        Returns:
            Plotly Figure with overlaid spray patterns
        """
        fig = go.Figure()

        colors = self.theme.get_color_sequence()
        cone_length = 50.0  # Standard visualization length

        for i, (config, label) in enumerate(zip(configurations, labels)):
            angle_rad = np.radians(config['spray_half_angle'])
            r_o = config['orifice_diameter'] / 2

            # Spray cone boundaries
            x_cone = np.array([0, cone_length])
            y_upper = np.array([r_o, r_o + cone_length * np.tan(angle_rad)])
            y_lower = np.array([-r_o, -r_o - cone_length * np.tan(angle_rad)])

            # Upper boundary
            fig.add_trace(go.Scatter(
                x=x_cone,
                y=y_upper,
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2),
                name=f'{label} (2a={2*config["spray_half_angle"]:.0f} deg)',
                hovertemplate=f"{label}<br>Angle: {config['spray_half_angle']:.1f} deg<extra></extra>"
            ))

            # Lower boundary
            fig.add_trace(go.Scatter(
                x=x_cone,
                y=y_lower,
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Centerline
        fig.add_trace(go.Scatter(
            x=[0, cone_length * 1.1],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', dash='dashdot', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))

        max_angle = max(np.radians(c['spray_half_angle']) for c in configurations)
        max_r_o = max(c['orifice_diameter'] / 2 for c in configurations)
        max_r = max_r_o + cone_length * np.tan(max_angle)

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis=dict(
                title="Axial Distance [mm]",
                showgrid=True,
                range=[-5, cone_length * 1.1]
            ),
            yaxis=dict(
                title="Radial Position [mm]",
                showgrid=True,
                scaleanchor="x",
                scaleratio=1,
                range=[-max_r * 1.2, max_r * 1.2]
            ),
            showlegend=True,
            legend=dict(x=1.02, y=0.98),
            height=500,
            width=800
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
