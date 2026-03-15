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


class ParameterStudyPlotter:
    """
    Runs CEA sweeps and produces a 2×2 parametric study figure.

    Sweeps:
    1. Isp_vac & C* vs O/F ratio at design Pc
    2. Isp_vac vs Chamber Pressure at design O/F
    3. Vacuum Isp vs Expansion Ratio at design Pc + MR
    4. CF_vac & CF_SL vs Expansion Ratio at design Pc + MR

    Example:
        plotter = ParameterStudyPlotter(theme=DarkTheme())
        fig = plotter.run_and_plot(fuel, oxidizer, pc, mr, eps)
        html = fig.to_json()
    """

    def __init__(self, theme=None):
        from resa.visualization.themes import DEFAULT_THEME
        self.theme = theme or DEFAULT_THEME

    # ------------------------------------------------------------------
    # Sweep helpers
    # ------------------------------------------------------------------

    def _mr_sweep(self, fuel: str, oxidizer: str, pc: float, eps: float,
                  mr_design: float, n: int = 35):
        """Sweep mixture ratio, return (mrs, isp_vac, cstar)."""
        from resa.solvers.combustion import CEASolver
        solver = CEASolver(fuel, oxidizer)
        mr_min = max(0.5, mr_design * 0.35)
        mr_max = mr_design * 2.8
        mrs = np.linspace(mr_min, mr_max, n)
        isps, cstars = [], []
        for mr in mrs:
            try:
                r = solver.run(pc, float(mr), eps)
                isps.append(r.isp_vac)
                cstars.append(r.cstar)
            except Exception:
                isps.append(np.nan)
                cstars.append(np.nan)
        return mrs, np.array(isps), np.array(cstars)

    def _pc_sweep(self, fuel: str, oxidizer: str, mr: float, eps: float,
                  pc_design: float, n: int = 30):
        """Sweep chamber pressure, return (pcs, isp_vac)."""
        from resa.solvers.combustion import CEASolver
        solver = CEASolver(fuel, oxidizer)
        pc_min = max(5.0, pc_design * 0.15)
        pc_max = max(200.0, pc_design * 3.5)
        pcs = np.linspace(pc_min, pc_max, n)
        isps = []
        for pc in pcs:
            try:
                r = solver.run(float(pc), mr, eps)
                isps.append(r.isp_vac)
            except Exception:
                isps.append(np.nan)
        return pcs, np.array(isps)

    def _eps_sweep(self, fuel: str, oxidizer: str, pc: float, mr: float,
                   n: int = 40):
        """Sweep expansion ratio, return (epss, isp_vac, cf_vac, cf_sl)."""
        from resa.solvers.combustion import CEASolver
        solver = CEASolver(fuel, oxidizer)
        epss = np.concatenate([
            np.linspace(1.5, 10.0, 20),
            np.linspace(10.5, 60.0, 20),
        ])
        isp_vac, cf_vac, cf_sl = [], [], []
        p_amb = 1.013  # bar sea-level
        g0 = 9.80665
        for eps in epss:
            try:
                r = solver.run(pc, mr, float(eps))
                isp_vac.append(r.isp_vac)
                cf_v = r.isp_vac * g0 / r.cstar if r.cstar > 0 else np.nan
                M_e = r.mach_exit
                gamma = getattr(r, 'gamma', 1.2)
                p_ratio = (1 + (gamma - 1) / 2 * M_e ** 2) ** (-gamma / (gamma - 1))
                p_exit_bar = pc * p_ratio
                cf_sl_val = cf_v - float(eps) * (p_amb - p_exit_bar) / pc
                cf_vac.append(cf_v)
                cf_sl.append(cf_sl_val)
            except Exception:
                isp_vac.append(np.nan)
                cf_vac.append(np.nan)
                cf_sl.append(np.nan)
        return epss, np.array(isp_vac), np.array(cf_vac), np.array(cf_sl)

    # ------------------------------------------------------------------
    # Main figure builder
    # ------------------------------------------------------------------

    def run_and_plot(
        self,
        fuel: str,
        oxidizer: str,
        pc: float,
        mr: float,
        eps: float,
    ) -> go.Figure:
        """
        Run all sweeps and return a 2×2 Plotly figure.

        Args:
            fuel: Fuel name (e.g. 'Ethanol90')
            oxidizer: Oxidizer name (e.g. 'N2O')
            pc: Design chamber pressure [bar]
            mr: Design mixture ratio (O/F)
            eps: Design expansion ratio (0 → use 6.0 default)

        Returns:
            Plotly Figure with 4 parametric study panels
        """
        eps_design = eps if eps and eps > 1.0 else 6.0

        mrs, isp_mr, cstar_mr = self._mr_sweep(fuel, oxidizer, pc, eps_design, mr)
        pcs, isp_pc = self._pc_sweep(fuel, oxidizer, mr, eps_design, pc)
        epss, isp_eps, cf_vac_eps, cf_sl_eps = self._eps_sweep(fuel, oxidizer, pc, mr)

        try:
            from resa.solvers.combustion import CEASolver
            dp = CEASolver(fuel, oxidizer).run(pc, mr, eps_design)
            dp_isp = dp.isp_vac
        except Exception:
            dp_isp = float(np.nanmean(isp_mr))

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"Isp & C* vs O/F  (Pc = {pc:.0f} bar)",
                f"Isp vs Chamber Pressure  (O/F = {mr:.2f})",
                f"Vacuum Isp vs Expansion Ratio  (Pc = {pc:.0f} bar, O/F = {mr:.2f})",
                "Thrust Coefficient CF vs Expansion Ratio",
            ),
            specs=[
                [{"secondary_y": True}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": True}],
            ],
            horizontal_spacing=0.14,
            vertical_spacing=0.18,
        )

        accent = self.theme.accent
        primary = self.theme.primary
        secondary = self.theme.secondary
        danger = self.theme.danger

        # Panel 1: Isp & C* vs O/F
        fig.add_trace(go.Scatter(
            x=mrs, y=isp_mr, name='Isp_vac [s]', mode='lines',
            line=dict(color=accent, width=2),
            hovertemplate='O/F=%{x:.2f}<br>Isp=%{y:.1f} s<extra></extra>',
        ), row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=mrs, y=cstar_mr, name='C* [m/s]', mode='lines',
            line=dict(color=secondary, width=2, dash='dash'),
            hovertemplate='O/F=%{x:.2f}<br>C*=%{y:.0f} m/s<extra></extra>',
        ), row=1, col=1, secondary_y=True)

        fig.add_trace(go.Scatter(
            x=[mr], y=[dp_isp], mode='markers', name='Design point',
            marker=dict(size=10, color='gold', symbol='star', line=dict(color='black', width=1)),
            hovertemplate=f'Design: O/F={mr:.2f}, Isp={dp_isp:.1f} s<extra></extra>',
        ), row=1, col=1, secondary_y=False)

        # Panel 2: Isp vs Pc
        fig.add_trace(go.Scatter(
            x=pcs, y=isp_pc, mode='lines', showlegend=False,
            line=dict(color=primary, width=2),
            hovertemplate='Pc=%{x:.1f} bar<br>Isp=%{y:.1f} s<extra></extra>',
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=[pc], y=[dp_isp], mode='markers', showlegend=False,
            marker=dict(size=10, color='gold', symbol='star', line=dict(color='black', width=1)),
            hovertemplate=f'Design: Pc={pc:.1f} bar, Isp={dp_isp:.1f} s<extra></extra>',
        ), row=1, col=2)

        # Panel 3: Isp vs eps
        fig.add_trace(go.Scatter(
            x=epss, y=isp_eps, mode='lines', showlegend=False,
            line=dict(color=accent, width=2),
            hovertemplate='ε=%{x:.1f}<br>Isp=%{y:.1f} s<extra></extra>',
        ), row=2, col=1)

        isp_at_design = float(np.interp(eps_design, epss, np.nan_to_num(isp_eps)))
        fig.add_trace(go.Scatter(
            x=[eps_design], y=[isp_at_design], mode='markers', showlegend=False,
            marker=dict(size=10, color='gold', symbol='star', line=dict(color='black', width=1)),
            hovertemplate=f'Design: ε={eps_design:.1f}, Isp={isp_at_design:.1f} s<extra></extra>',
        ), row=2, col=1)

        # Panel 4: CF vs eps
        fig.add_trace(go.Scatter(
            x=epss, y=cf_vac_eps, name='CF_vac', mode='lines', showlegend=False,
            line=dict(color=accent, width=2),
            hovertemplate='ε=%{x:.1f}<br>CF_vac=%{y:.3f}<extra></extra>',
        ), row=2, col=2, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=epss, y=cf_sl_eps, name='CF_SL', mode='lines', showlegend=False,
            line=dict(color=danger, width=2, dash='dash'),
            hovertemplate='ε=%{x:.1f}<br>CF_SL=%{y:.3f}<extra></extra>',
        ), row=2, col=2, secondary_y=True)

        cf_v_design = float(np.interp(eps_design, epss, np.nan_to_num(cf_vac_eps)))
        fig.add_trace(go.Scatter(
            x=[eps_design], y=[cf_v_design], mode='markers', showlegend=False,
            marker=dict(size=10, color='gold', symbol='star', line=dict(color='black', width=1)),
            hoverinfo='skip',
        ), row=2, col=2, secondary_y=False)

        # Axis labels
        fig.update_xaxes(title_text="O/F Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Isp_vac [s]", row=1, col=1, secondary_y=False,
                         title_font=dict(color=accent))
        fig.update_yaxes(title_text="C* [m/s]", row=1, col=1, secondary_y=True,
                         title_font=dict(color=secondary))

        fig.update_xaxes(title_text="Chamber Pressure [bar]", row=1, col=2)
        fig.update_yaxes(title_text="Isp_vac [s]", row=1, col=2)

        fig.update_xaxes(title_text="Expansion Ratio ε", row=2, col=1)
        fig.update_yaxes(title_text="Isp_vac [s]", row=2, col=1)

        fig.update_xaxes(title_text="Expansion Ratio ε", row=2, col=2)
        fig.update_yaxes(title_text="CF_vac", row=2, col=2, secondary_y=False,
                         title_font=dict(color=accent))
        fig.update_yaxes(title_text="CF_SL", row=2, col=2, secondary_y=True,
                         title_font=dict(color=danger))

        fig.update_layout(
            height=680,
            title=dict(
                text=f"Parameter Study  —  {fuel} / {oxidizer}",
                x=0.5, font=dict(size=14),
            ),
            showlegend=True,
            legend=dict(orientation='h', y=-0.06, x=0.5, xanchor='center', font=dict(size=10)),
            hovermode='x unified',
            margin=dict(l=55, r=55, t=70, b=60),
        )
        self.theme.apply_to_figure(fig)
        return fig

    def to_html(self, fig: go.Figure, include_plotlyjs: str = 'cdn', full_html: bool = False) -> str:
        return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=full_html)


class OperatingEnvelopePlotter:
    """
    Visualizes engine operating envelope with constraints.

    Creates interactive plots showing:
    - Valid operating region (Pc vs MR)
    - Thermal constraint boundaries
    - Stability limits
    - Performance contours within envelope

    Example:
        plotter = OperatingEnvelopePlotter()
        fig = plotter.create_figure(envelope_data)
        fig.show()
    """

    def __init__(self, theme: Optional[PlotTheme] = None):
        """Initialize with optional custom theme."""
        self.theme = theme or DEFAULT_THEME

    def create_figure(
        self,
        pc_values: np.ndarray,
        mr_values: np.ndarray,
        performance_grid: np.ndarray,
        constraints: Optional[Dict[str, np.ndarray]] = None,
        design_point: Optional[Dict[str, float]] = None,
        title: str = "Engine Operating Envelope"
    ) -> go.Figure:
        """
        Create operating envelope visualization.

        Args:
            pc_values: 1D array of chamber pressures [bar]
            mr_values: 1D array of mixture ratios
            performance_grid: 2D array of performance metric (e.g., Isp)
            constraints: Dict with constraint boundaries:
                - 'thermal_limit': 2D boolean array (True = valid)
                - 'stability_limit': 2D boolean array
                - 'pressure_limit': 2D boolean array
            design_point: Optional dict with 'pc' and 'mr' keys
            title: Plot title

        Returns:
            Plotly Figure with operating envelope
        """
        fig = go.Figure()

        # ========== Performance Contour (base layer) ==========
        fig.add_trace(go.Contour(
            x=pc_values,
            y=mr_values,
            z=performance_grid,
            colorscale='Viridis',
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='white'),
            ),
            colorbar=dict(
                title='Isp [s]',
                titleside='right',
                x=1.02
            ),
            name='Performance',
            hovertemplate="Pc: %{x:.1f} bar<br>O/F: %{y:.2f}<br>Isp: %{z:.0f} s<extra></extra>"
        ))

        # ========== Constraint Boundaries ==========
        if constraints:
            # Thermal limit (typically max wall temperature)
            if 'thermal_limit' in constraints:
                thermal_valid = constraints['thermal_limit']
                # Create boundary contour
                fig.add_trace(go.Contour(
                    x=pc_values,
                    y=mr_values,
                    z=thermal_valid.astype(float),
                    contours=dict(
                        start=0.5, end=0.5, size=0,
                        coloring='none',
                    ),
                    line=dict(color=self.theme.danger, width=3, dash='dash'),
                    showscale=False,
                    name='Thermal Limit',
                    hoverinfo='skip'
                ))

            # Stability limit (e.g., combustion stability)
            if 'stability_limit' in constraints:
                stability_valid = constraints['stability_limit']
                fig.add_trace(go.Contour(
                    x=pc_values,
                    y=mr_values,
                    z=stability_valid.astype(float),
                    contours=dict(
                        start=0.5, end=0.5, size=0,
                        coloring='none',
                    ),
                    line=dict(color=self.theme.secondary, width=3, dash='dot'),
                    showscale=False,
                    name='Stability Limit',
                    hoverinfo='skip'
                ))

            # Pressure limit (e.g., feed system capability)
            if 'pressure_limit' in constraints:
                pressure_valid = constraints['pressure_limit']
                fig.add_trace(go.Contour(
                    x=pc_values,
                    y=mr_values,
                    z=pressure_valid.astype(float),
                    contours=dict(
                        start=0.5, end=0.5, size=0,
                        coloring='none',
                    ),
                    line=dict(color=self.theme.info, width=3),
                    showscale=False,
                    name='Pressure Limit',
                    hoverinfo='skip'
                ))

            # Shade invalid regions
            combined_valid = np.ones_like(performance_grid, dtype=bool)
            for key, valid in constraints.items():
                combined_valid &= valid

            invalid_mask = ~combined_valid
            if invalid_mask.any():
                # Create semi-transparent overlay for invalid regions
                invalid_z = np.where(invalid_mask, 1.0, np.nan)
                fig.add_trace(go.Heatmap(
                    x=pc_values,
                    y=mr_values,
                    z=invalid_z,
                    colorscale=[[0, 'rgba(128,128,128,0.4)'], [1, 'rgba(128,128,128,0.4)']],
                    showscale=False,
                    hoverinfo='skip',
                    name='Invalid Region'
                ))

        # ========== Design Point ==========
        if design_point:
            fig.add_trace(go.Scatter(
                x=[design_point['pc']],
                y=[design_point['mr']],
                mode='markers',
                marker=dict(
                    size=20,
                    symbol='star',
                    color='gold',
                    line=dict(color='black', width=2)
                ),
                name='Design Point',
                hovertemplate=(
                    f"Design Point<br>"
                    f"Pc: {design_point['pc']:.1f} bar<br>"
                    f"O/F: {design_point['mr']:.2f}<extra></extra>"
                )
            ))

        # ========== Throttle Range Indicators ==========
        if design_point and 'throttle_range' in design_point:
            throttle_min = design_point.get('throttle_min', 0.3)
            throttle_max = design_point.get('throttle_max', 1.0)
            pc_design = design_point['pc']

            # Draw throttle range line
            fig.add_trace(go.Scatter(
                x=[pc_design * throttle_min, pc_design * throttle_max],
                y=[design_point['mr'], design_point['mr']],
                mode='lines+markers',
                line=dict(color='white', width=3),
                marker=dict(size=10, symbol='line-ns', color='white'),
                name=f'Throttle Range ({throttle_min*100:.0f}-{throttle_max*100:.0f}%)',
                hoverinfo='skip'
            ))

        # ========== Layout ==========
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            xaxis=dict(
                title="Chamber Pressure [bar]",
                showgrid=True,
                gridcolor=self.theme.grid_color
            ),
            yaxis=dict(
                title="Mixture Ratio (O/F)",
                showgrid=True,
                gridcolor=self.theme.grid_color
            ),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='black',
                borderwidth=1
            ),
            width=900,
            height=700,
            hovermode='closest'
        )

        self.theme.apply_to_figure(fig)

        return fig

    def create_multi_metric_envelope(
        self,
        pc_values: np.ndarray,
        mr_values: np.ndarray,
        metrics: Dict[str, np.ndarray],
        design_point: Optional[Dict[str, float]] = None,
        title: str = "Multi-Metric Operating Envelope"
    ) -> go.Figure:
        """
        Create envelope visualization with multiple performance metrics.

        Args:
            pc_values: 1D array of chamber pressures [bar]
            mr_values: 1D array of mixture ratios
            metrics: Dict of 2D arrays with keys like 'isp', 'cstar', 'thrust'
            design_point: Optional dict with 'pc' and 'mr' keys
            title: Plot title

        Returns:
            Plotly Figure with subplots for each metric
        """
        n_metrics = len(metrics)
        cols = min(2, n_metrics)
        rows = (n_metrics + 1) // 2

        subplot_titles = [name.upper() for name in metrics.keys()]

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.15,
            vertical_spacing=0.12
        )

        colorscales = ['Viridis', 'Plasma', 'Inferno', 'Cividis']

        for idx, (name, data) in enumerate(metrics.items()):
            row = idx // cols + 1
            col = idx % cols + 1

            fig.add_trace(
                go.Contour(
                    x=pc_values,
                    y=mr_values,
                    z=data,
                    colorscale=colorscales[idx % len(colorscales)],
                    contours=dict(showlabels=True),
                    colorbar=dict(
                        title=name,
                        len=0.4,
                        y=1 - (row - 0.5) / rows,
                        x=col / cols + 0.02 if col == cols else None
                    ),
                    showscale=True,
                    hovertemplate=f"Pc: %{{x:.1f}}<br>O/F: %{{y:.2f}}<br>{name}: %{{z:.1f}}<extra></extra>"
                ),
                row=row, col=col
            )

            # Add design point to each subplot
            if design_point:
                fig.add_trace(
                    go.Scatter(
                        x=[design_point['pc']],
                        y=[design_point['mr']],
                        mode='markers',
                        marker=dict(size=12, symbol='x', color='red', line=dict(width=2)),
                        showlegend=(idx == 0),
                        name='Design Point',
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )

            fig.update_xaxes(title_text="Pc [bar]", row=row, col=col)
            fig.update_yaxes(title_text="O/F", row=row, col=col)

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            height=400 * rows,
            width=500 * cols,
            showlegend=True
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
