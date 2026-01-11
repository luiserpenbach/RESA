"""
Igniter visualization module using Plotly.

Provides interactive visualizations for:
- Igniter chamber/nozzle schematics
- Mixture ratio sweep results
- Performance envelopes
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
from dataclasses import dataclass

from resa.visualization.themes import PlotTheme, EngineeringTheme, DEFAULT_THEME

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class IgniterGeometry:
    """
    Igniter geometry data container.

    Attributes:
        chamber_diameter: Chamber diameter [m]
        chamber_length: Chamber length [m]
        throat_diameter: Throat diameter [m]
        exit_diameter: Exit/nozzle exit diameter [m]
        nozzle_length: Nozzle length [m]
        contraction_angle: Convergent section half angle [deg]
        expansion_angle: Divergent section half angle [deg]
    """
    chamber_diameter: float
    chamber_length: float
    throat_diameter: float
    exit_diameter: float
    nozzle_length: float = None
    contraction_angle: float = 45.0
    expansion_angle: float = 15.0


class IgniterSchematicPlotter:
    """
    Creates schematic visualization of igniter geometry.

    Shows:
    - Chamber section
    - Convergent nozzle section
    - Throat
    - Divergent nozzle section
    - Key dimensions

    Example:
        plotter = IgniterSchematicPlotter()
        fig = plotter.create_figure(geometry)
        fig.show()
    """

    def __init__(self, theme: Optional[PlotTheme] = None):
        """Initialize with optional custom theme."""
        self.theme = theme or DEFAULT_THEME

    def create_figure(
        self,
        geometry: IgniterGeometry,
        show_dimensions: bool = True,
        show_flow_regions: bool = True,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create igniter geometry schematic.

        Args:
            geometry: IgniterGeometry dataclass with dimensions
            show_dimensions: If True, shows dimension annotations
            show_flow_regions: If True, shows combustion/flow regions
            title: Optional custom title

        Returns:
            Plotly Figure with igniter schematic
        """
        fig = go.Figure()

        # Convert to mm
        D_ch = geometry.chamber_diameter * 1000
        L_ch = geometry.chamber_length * 1000
        D_t = geometry.throat_diameter * 1000
        D_e = geometry.exit_diameter * 1000

        R_ch = D_ch / 2
        R_t = D_t / 2
        R_e = D_e / 2

        # Calculate nozzle geometry
        conv_angle_rad = np.radians(geometry.contraction_angle)
        exp_angle_rad = np.radians(geometry.expansion_angle)

        # Convergent length
        L_conv = (R_ch - R_t) / np.tan(conv_angle_rad)

        # Divergent length (use provided or calculate)
        if geometry.nozzle_length:
            L_nozzle = geometry.nozzle_length * 1000
            L_div = L_nozzle - L_conv
        else:
            L_div = (R_e - R_t) / np.tan(exp_angle_rad)
            L_nozzle = L_conv + L_div

        # X coordinates
        x_chamber_start = 0
        x_chamber_end = L_ch
        x_throat = L_ch + L_conv
        x_exit = x_throat + L_div

        # ========== Chamber Section ==========
        chamber_x = [x_chamber_start, x_chamber_end, x_chamber_end, x_chamber_start, x_chamber_start]
        chamber_y_upper = [R_ch, R_ch, 0, 0, R_ch]
        chamber_y_lower = [-R_ch, -R_ch, 0, 0, -R_ch]

        # Upper chamber
        fig.add_trace(go.Scatter(
            x=chamber_x,
            y=chamber_y_upper,
            fill='toself',
            fillcolor='rgba(135, 206, 235, 0.3)',  # Light blue
            line=dict(color='black', width=2),
            name='Chamber',
            hovertemplate=f"Chamber<br>D = {D_ch:.1f} mm<br>L = {L_ch:.1f} mm<extra></extra>"
        ))

        # Lower chamber (mirror)
        fig.add_trace(go.Scatter(
            x=chamber_x,
            y=chamber_y_lower,
            fill='toself',
            fillcolor='rgba(135, 206, 235, 0.3)',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

        # ========== Convergent Section ==========
        conv_x = [x_chamber_end, x_throat, x_throat, x_chamber_end, x_chamber_end]
        conv_y_upper = [R_ch, R_t, 0, 0, R_ch]
        conv_y_lower = [-R_ch, -R_t, 0, 0, -R_ch]

        fig.add_trace(go.Scatter(
            x=conv_x,
            y=conv_y_upper,
            fill='toself',
            fillcolor='rgba(255, 200, 100, 0.4)',  # Light orange
            line=dict(color='black', width=2),
            name='Convergent',
            hovertemplate=f"Convergent Section<br>Angle: {geometry.contraction_angle:.0f} deg<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=conv_x,
            y=conv_y_lower,
            fill='toself',
            fillcolor='rgba(255, 200, 100, 0.4)',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

        # ========== Divergent Section ==========
        div_x = [x_throat, x_exit, x_exit, x_throat, x_throat]
        div_y_upper = [R_t, R_e, 0, 0, R_t]
        div_y_lower = [-R_t, -R_e, 0, 0, -R_t]

        fig.add_trace(go.Scatter(
            x=div_x,
            y=div_y_upper,
            fill='toself',
            fillcolor='rgba(255, 100, 100, 0.3)',  # Light red
            line=dict(color='black', width=2),
            name='Divergent',
            hovertemplate=f"Divergent Section<br>Angle: {geometry.expansion_angle:.0f} deg<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=div_x,
            y=div_y_lower,
            fill='toself',
            fillcolor='rgba(255, 100, 100, 0.3)',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

        # ========== Throat Indicator ==========
        fig.add_trace(go.Scatter(
            x=[x_throat, x_throat],
            y=[-R_t * 1.3, R_t * 1.3],
            mode='lines',
            line=dict(color=self.theme.danger, width=2, dash='dash'),
            name='Throat',
            hovertemplate=f"Throat: D* = {D_t:.2f} mm<extra></extra>"
        ))

        # ========== Centerline ==========
        fig.add_trace(go.Scatter(
            x=[x_chamber_start - 5, x_exit + 10],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', dash='dashdot', width=1),
            name='Centerline',
            showlegend=False,
            hoverinfo='skip'
        ))

        # ========== Flow Regions (flame visualization) ==========
        if show_flow_regions:
            # Combustion zone (flame)
            flame_x = np.linspace(L_ch * 0.1, L_ch * 0.8, 50)
            flame_upper = R_ch * 0.7 * np.sin(np.pi * (flame_x - L_ch * 0.1) / (L_ch * 0.7))
            flame_lower = -flame_upper

            fig.add_trace(go.Scatter(
                x=np.concatenate([flame_x, flame_x[::-1]]),
                y=np.concatenate([flame_upper, flame_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 165, 0, 0.4)',  # Orange for flame
                line=dict(color='orange', width=1),
                name='Combustion Zone',
                hoverinfo='skip'
            ))

            # Hot gas core through nozzle
            core_x = np.array([L_ch * 0.8, x_throat, x_exit])
            core_y_upper = np.array([R_ch * 0.5, R_t * 0.8, R_e * 0.7])
            core_y_lower = -core_y_upper

            fig.add_trace(go.Scatter(
                x=np.concatenate([core_x, core_x[::-1]]),
                y=np.concatenate([core_y_upper, core_y_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 69, 0, 0.3)',  # Red-orange
                line=dict(width=0),
                name='Hot Gas',
                showlegend=False,
                hoverinfo='skip'
            ))

        # ========== Dimension Annotations ==========
        if show_dimensions:
            annotations = [
                # Chamber diameter
                dict(
                    x=L_ch / 2, y=R_ch + 3,
                    text=f"D_ch = {D_ch:.1f} mm",
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                # Chamber length
                dict(
                    x=L_ch / 2, y=-R_ch - 5,
                    text=f"L_ch = {L_ch:.1f} mm",
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                # Throat diameter
                dict(
                    x=x_throat, y=R_t + 3,
                    text=f"D* = {D_t:.2f} mm",
                    showarrow=True, arrowhead=2, ax=0, ay=-25,
                    font=dict(size=10, color=self.theme.danger),
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                # Exit diameter
                dict(
                    x=x_exit, y=R_e + 3,
                    text=f"D_e = {D_e:.1f} mm",
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                # Expansion ratio
                dict(
                    x=x_throat + L_div / 2, y=-R_e - 5,
                    text=f"AR = {(D_e/D_t)**2:.1f}",
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor='rgba(255,255,255,0.8)'
                ),
            ]

            for ann in annotations:
                fig.add_annotation(**ann)

        # ========== Layout ==========
        max_y = max(R_ch, R_e) * 1.5

        fig.update_layout(
            title=dict(
                text=title or "Torch Igniter Geometry",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis=dict(
                title="Axial Position [mm]",
                scaleanchor="y",
                scaleratio=1,
                showgrid=True,
                gridcolor=self.theme.grid_color,
                range=[-10, x_exit + 15]
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
            width=900,
            margin=dict(l=60, r=150, t=80, b=60)
        )

        self.theme.apply_to_figure(fig)

        return fig

    def create_with_injector(
        self,
        geometry: IgniterGeometry,
        injector_config: Optional[Dict[str, float]] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create igniter schematic with injector elements indicated.

        Args:
            geometry: IgniterGeometry dataclass
            injector_config: Dict with 'ox_diameter', 'fuel_diameter' in mm
            title: Optional custom title

        Returns:
            Plotly Figure with igniter and injector
        """
        # Create base schematic
        fig = self.create_figure(geometry, show_dimensions=True, show_flow_regions=False, title=title)

        # Convert dimensions
        D_ch = geometry.chamber_diameter * 1000
        R_ch = D_ch / 2

        # Add injector elements at head end
        if injector_config:
            ox_d = injector_config.get('ox_diameter', 2.0)
            fuel_d = injector_config.get('fuel_diameter', 1.5)

            # Oxidizer injection (center)
            fig.add_shape(
                type="circle",
                x0=-ox_d / 2, y0=-ox_d / 2,
                x1=ox_d / 2, y1=ox_d / 2,
                line=dict(color=self.theme.oxidizer, width=2),
                fillcolor=self.theme.oxidizer,
                opacity=0.5
            )

            # Fuel injection (coaxial ring - simplified as dots)
            fuel_positions = np.linspace(-R_ch * 0.5, R_ch * 0.5, 4)
            for y_pos in fuel_positions:
                if abs(y_pos) > ox_d:
                    fig.add_shape(
                        type="circle",
                        x0=-fuel_d / 2, y0=y_pos - fuel_d / 2,
                        x1=fuel_d / 2, y1=y_pos + fuel_d / 2,
                        line=dict(color=self.theme.fuel, width=1),
                        fillcolor=self.theme.fuel,
                        opacity=0.5
                    )

            # Add legend entries
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=self.theme.oxidizer),
                name=f'Ox Injection (D={ox_d:.1f}mm)'
            ))
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=self.theme.fuel),
                name=f'Fuel Injection (D={fuel_d:.1f}mm)'
            ))

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


class MixtureSweepPlotter:
    """
    Visualizes mixture ratio sweep results for igniter performance.

    Creates multi-panel plots showing:
    - C* and Isp vs MR
    - Flame temperature
    - Thrust and heat power
    - Mass flows

    Example:
        plotter = MixtureSweepPlotter()
        fig = plotter.create_figure(sweep_data)
        fig.show()
    """

    def __init__(self, theme: Optional[PlotTheme] = None):
        """Initialize with optional custom theme."""
        self.theme = theme or DEFAULT_THEME

    def create_figure(
        self,
        sweep_data: Union[List[Dict[str, float]], 'pd.DataFrame'],
        design_point: Optional[Dict[str, float]] = None,
        title: str = "Igniter Performance vs. Mixture Ratio"
    ) -> go.Figure:
        """
        Create mixture ratio sweep visualization.

        Args:
            sweep_data: List of dicts or DataFrame with columns:
                - 'mixture_ratio' or 'mr': O/F ratio
                - 'c_star': Characteristic velocity [m/s]
                - 'isp': Specific impulse [s]
                - 'flame_temperature' or 'T_flame': Flame temp [K]
                - 'thrust': Thrust [N] (optional)
                - 'heat_power': Heat power [W] (optional)
            design_point: Optional dict with 'mr' key to highlight
            title: Plot title

        Returns:
            Plotly Figure with sweep results
        """
        # Convert to list of dicts if DataFrame
        if hasattr(sweep_data, 'to_dict'):
            data = sweep_data.to_dict('records')
        else:
            data = sweep_data

        # Extract data arrays
        mr_key = 'mixture_ratio' if 'mixture_ratio' in data[0] else 'mr'
        T_key = 'flame_temperature' if 'flame_temperature' in data[0] else 'T_flame'

        mrs = [d[mr_key] for d in data]
        c_stars = [d['c_star'] for d in data]
        isps = [d['isp'] for d in data]
        T_flames = [d[T_key] for d in data]

        has_thrust = 'thrust' in data[0]
        has_heat = 'heat_power' in data[0]

        # Determine subplot layout
        n_rows = 2
        if has_thrust or has_heat:
            n_rows = 2

        fig = make_subplots(
            rows=n_rows, cols=2,
            subplot_titles=(
                "C* and Isp",
                "Flame Temperature",
                "Thrust and Heat Power" if has_thrust else "Gamma",
                "Mass Flows" if 'n2o_mass_flow' in data[0] else "Performance Index"
            ),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )

        # ========== Panel 1: C* and Isp ==========
        fig.add_trace(
            go.Scatter(
                x=mrs, y=c_stars,
                mode='lines+markers',
                name='C*',
                line=dict(color=self.theme.primary, width=2),
                marker=dict(size=6),
                hovertemplate="O/F: %{x:.2f}<br>C*: %{y:.0f} m/s<extra></extra>"
            ),
            row=1, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=mrs, y=isps,
                mode='lines+markers',
                name='Isp',
                line=dict(color=self.theme.secondary, width=2),
                marker=dict(size=6),
                hovertemplate="O/F: %{x:.2f}<br>Isp: %{y:.1f} s<extra></extra>"
            ),
            row=1, col=1, secondary_y=True
        )

        # ========== Panel 2: Flame Temperature ==========
        fig.add_trace(
            go.Scatter(
                x=mrs, y=T_flames,
                mode='lines+markers',
                name='T_flame',
                line=dict(color=self.theme.danger, width=2),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.1)',
                hovertemplate="O/F: %{x:.2f}<br>T: %{y:.0f} K<extra></extra>"
            ),
            row=1, col=2
        )

        # ========== Panel 3: Thrust and Heat Power ==========
        if has_thrust:
            thrusts = [d['thrust'] for d in data]
            fig.add_trace(
                go.Scatter(
                    x=mrs, y=thrusts,
                    mode='lines+markers',
                    name='Thrust',
                    line=dict(color=self.theme.accent, width=2),
                    marker=dict(size=6),
                    hovertemplate="O/F: %{x:.2f}<br>F: %{y:.1f} N<extra></extra>"
                ),
                row=2, col=1, secondary_y=False
            )

        if has_heat:
            heat_powers = [d['heat_power'] / 1000 for d in data]  # Convert to kW
            fig.add_trace(
                go.Scatter(
                    x=mrs, y=heat_powers,
                    mode='lines+markers',
                    name='Heat Power',
                    line=dict(color='purple', width=2),
                    marker=dict(size=6),
                    hovertemplate="O/F: %{x:.2f}<br>Q: %{y:.1f} kW<extra></extra>"
                ),
                row=2, col=1, secondary_y=True
            )

        if not has_thrust and 'gamma' in data[0]:
            gammas = [d['gamma'] for d in data]
            fig.add_trace(
                go.Scatter(
                    x=mrs, y=gammas,
                    mode='lines+markers',
                    name='Gamma',
                    line=dict(color=self.theme.info, width=2),
                    marker=dict(size=6),
                    hovertemplate="O/F: %{x:.2f}<br>gamma: %{y:.3f}<extra></extra>"
                ),
                row=2, col=1, secondary_y=False
            )

        # ========== Panel 4: Mass Flows or Performance Index ==========
        if 'n2o_mass_flow' in data[0]:
            n2o_flows = [d['n2o_mass_flow'] * 1000 for d in data]  # g/s
            fuel_flows = [d.get('ethanol_mass_flow', d.get('fuel_mass_flow', 0)) * 1000 for d in data]

            fig.add_trace(
                go.Scatter(
                    x=mrs, y=n2o_flows,
                    mode='lines+markers',
                    name='N2O',
                    line=dict(color=self.theme.oxidizer, width=2),
                    marker=dict(size=6),
                    hovertemplate="O/F: %{x:.2f}<br>N2O: %{y:.1f} g/s<extra></extra>"
                ),
                row=2, col=2
            )

            fig.add_trace(
                go.Scatter(
                    x=mrs, y=fuel_flows,
                    mode='lines+markers',
                    name='Fuel',
                    line=dict(color=self.theme.fuel, width=2),
                    marker=dict(size=6),
                    hovertemplate="O/F: %{x:.2f}<br>Fuel: %{y:.1f} g/s<extra></extra>"
                ),
                row=2, col=2
            )
        else:
            # Performance index (C* * Isp normalized)
            c_star_max = max(c_stars)
            isp_max = max(isps)
            perf_index = [(c / c_star_max + i / isp_max) / 2 for c, i in zip(c_stars, isps)]

            fig.add_trace(
                go.Scatter(
                    x=mrs, y=perf_index,
                    mode='lines+markers',
                    name='Perf Index',
                    line=dict(color=self.theme.info, width=2),
                    marker=dict(size=6),
                    fill='tozeroy',
                    fillcolor='rgba(23, 162, 184, 0.1)',
                    hovertemplate="O/F: %{x:.2f}<br>Index: %{y:.3f}<extra></extra>"
                ),
                row=2, col=2
            )

        # ========== Design Point Markers ==========
        if design_point:
            mr_design = design_point.get('mr', design_point.get('mixture_ratio'))

            # Find closest point in data
            idx = min(range(len(mrs)), key=lambda i: abs(mrs[i] - mr_design))

            marker_style = dict(
                size=15,
                symbol='star',
                color='gold',
                line=dict(color='black', width=2)
            )

            # Add to each subplot
            fig.add_trace(
                go.Scatter(
                    x=[mr_design], y=[c_stars[idx]],
                    mode='markers',
                    marker=marker_style,
                    name='Design Point',
                    showlegend=True,
                    hovertemplate=f"Design<br>O/F: {mr_design:.2f}<extra></extra>"
                ),
                row=1, col=1, secondary_y=False
            )

            fig.add_trace(
                go.Scatter(
                    x=[mr_design], y=[T_flames[idx]],
                    mode='markers',
                    marker=marker_style,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=2
            )

        # ========== Axis Labels ==========
        fig.update_xaxes(title_text="Mixture Ratio (O/F)", row=1, col=1)
        fig.update_xaxes(title_text="Mixture Ratio (O/F)", row=1, col=2)
        fig.update_xaxes(title_text="Mixture Ratio (O/F)", row=2, col=1)
        fig.update_xaxes(title_text="Mixture Ratio (O/F)", row=2, col=2)

        fig.update_yaxes(title_text="C* [m/s]", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Isp [s]", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Temperature [K]", row=1, col=2)

        if has_thrust:
            fig.update_yaxes(title_text="Thrust [N]", row=2, col=1, secondary_y=False)
        else:
            fig.update_yaxes(title_text="Gamma [-]", row=2, col=1, secondary_y=False)

        if has_heat:
            fig.update_yaxes(title_text="Heat Power [kW]", row=2, col=1, secondary_y=True)

        if 'n2o_mass_flow' in data[0]:
            fig.update_yaxes(title_text="Mass Flow [g/s]", row=2, col=2)
        else:
            fig.update_yaxes(title_text="Normalized Index [-]", row=2, col=2)

        # ========== Layout ==========
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18)
            ),
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.12,
                xanchor="center",
                x=0.5
            ),
            hovermode='x unified'
        )

        self.theme.apply_to_figure(fig)

        return fig

    def create_pressure_sweep(
        self,
        sweep_data: Union[List[Dict[str, float]], 'pd.DataFrame'],
        design_point: Optional[Dict[str, float]] = None,
        title: str = "Igniter Performance vs. Chamber Pressure"
    ) -> go.Figure:
        """
        Create chamber pressure sweep visualization.

        Args:
            sweep_data: List of dicts or DataFrame with columns:
                - 'chamber_pressure' or 'pc': Chamber pressure [bar]
                - 'c_star', 'isp', 'flame_temperature', etc.
            design_point: Optional dict with 'pc' key to highlight
            title: Plot title

        Returns:
            Plotly Figure with pressure sweep results
        """
        # Convert to list of dicts if DataFrame
        if hasattr(sweep_data, 'to_dict'):
            data = sweep_data.to_dict('records')
        else:
            data = sweep_data

        # Extract data
        pc_key = 'chamber_pressure' if 'chamber_pressure' in data[0] else 'pc'
        T_key = 'flame_temperature' if 'flame_temperature' in data[0] else 'T_flame'

        pcs = [d[pc_key] for d in data]
        c_stars = [d['c_star'] for d in data]
        isps = [d['isp'] for d in data]
        T_flames = [d[T_key] for d in data]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "C* and Isp",
                "Flame Temperature",
                "Thrust" if 'thrust' in data[0] else "C* Efficiency",
                "Specific Impulse Detail"
            ),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )

        # Panel 1: C* and Isp
        fig.add_trace(
            go.Scatter(
                x=pcs, y=c_stars,
                mode='lines+markers',
                name='C*',
                line=dict(color=self.theme.primary, width=2),
                hovertemplate="Pc: %{x:.1f} bar<br>C*: %{y:.0f} m/s<extra></extra>"
            ),
            row=1, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=pcs, y=isps,
                mode='lines+markers',
                name='Isp',
                line=dict(color=self.theme.secondary, width=2),
                hovertemplate="Pc: %{x:.1f} bar<br>Isp: %{y:.1f} s<extra></extra>"
            ),
            row=1, col=1, secondary_y=True
        )

        # Panel 2: Flame Temperature
        fig.add_trace(
            go.Scatter(
                x=pcs, y=T_flames,
                mode='lines+markers',
                name='T_flame',
                line=dict(color=self.theme.danger, width=2),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.1)',
                hovertemplate="Pc: %{x:.1f} bar<br>T: %{y:.0f} K<extra></extra>"
            ),
            row=1, col=2
        )

        # Panel 3: Thrust or C* efficiency
        if 'thrust' in data[0]:
            thrusts = [d['thrust'] for d in data]
            fig.add_trace(
                go.Scatter(
                    x=pcs, y=thrusts,
                    mode='lines+markers',
                    name='Thrust',
                    line=dict(color=self.theme.accent, width=2),
                    hovertemplate="Pc: %{x:.1f} bar<br>F: %{y:.1f} N<extra></extra>"
                ),
                row=2, col=1
            )
        else:
            # Normalized C*
            c_star_ideal = max(c_stars) * 1.02
            c_star_eff = [c / c_star_ideal * 100 for c in c_stars]
            fig.add_trace(
                go.Scatter(
                    x=pcs, y=c_star_eff,
                    mode='lines+markers',
                    name='C* Efficiency',
                    line=dict(color=self.theme.accent, width=2),
                    hovertemplate="Pc: %{x:.1f} bar<br>Eff: %{y:.1f}%<extra></extra>"
                ),
                row=2, col=1
            )

        # Panel 4: Isp detail
        fig.add_trace(
            go.Scatter(
                x=pcs, y=isps,
                mode='lines+markers',
                name='Isp Detail',
                line=dict(color=self.theme.secondary, width=3),
                fill='tozeroy',
                fillcolor='rgba(255, 127, 14, 0.1)',
                showlegend=False,
                hovertemplate="Pc: %{x:.1f} bar<br>Isp: %{y:.1f} s<extra></extra>"
            ),
            row=2, col=2
        )

        # Design point markers
        if design_point:
            pc_design = design_point.get('pc', design_point.get('chamber_pressure'))
            idx = min(range(len(pcs)), key=lambda i: abs(pcs[i] - pc_design))

            marker_style = dict(size=15, symbol='star', color='gold', line=dict(color='black', width=2))

            fig.add_trace(
                go.Scatter(x=[pc_design], y=[c_stars[idx]], mode='markers',
                           marker=marker_style, name='Design Point'),
                row=1, col=1, secondary_y=False
            )

        # Axis labels
        for row in [1, 2]:
            for col in [1, 2]:
                fig.update_xaxes(title_text="Chamber Pressure [bar]", row=row, col=col)

        fig.update_yaxes(title_text="C* [m/s]", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Isp [s]", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Temperature [K]", row=1, col=2)
        fig.update_yaxes(title_text="Thrust [N]" if 'thrust' in data[0] else "C* Efficiency [%]", row=2, col=1)
        fig.update_yaxes(title_text="Isp [s]", row=2, col=2)

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            height=800,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
            hovermode='x unified'
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
