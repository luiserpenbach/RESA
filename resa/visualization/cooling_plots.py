"""
N2O Cooling Phase Diagram Visualizations.

Provides interactive Plotly phase diagrams for coolant state analysis:
- T-ρ (Temperature vs Density) diagram with pressure contour background
- P-T (Pressure vs Temperature) diagram with density contour background

Both diagrams overlay the coolant operating path, making it easy to identify
phase transitions, proximity to the critical point, and two-phase regions.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import plotly.graph_objects as go

from resa.visualization.themes import DEFAULT_THEME, PlotTheme

if TYPE_CHECKING:
    from resa.core.results import CoolingResult


class CoolingPhaseDiagramPlotter:
    """
    Creates interactive N2O phase diagrams with coolant operating paths.

    Supported diagrams:
    - T-ρ: Temperature vs Density, background shows pressure contours
    - P-T: Pressure vs Temperature, background shows density contours

    Example::

        plotter = CoolingPhaseDiagramPlotter(theme=DarkTheme())
        fig_trho = plotter.create_t_rho_figure(cooling_result)
        fig_pt   = plotter.create_p_t_figure(cooling_result)
    """

    # N2O critical constants
    _T_CRIT = 309.52   # K
    _P_CRIT = 72.45e5  # Pa
    _RHO_CRIT = 452.0  # kg/m³
    _T_TRIPLE = 182.33  # K (approximate)

    def __init__(self, theme: Optional[PlotTheme] = None):
        self.theme = theme or DEFAULT_THEME

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_t_rho_figure(
        self,
        cooling_result: "CoolingResult",
        fluid_name: str = "NitrousOxide",
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create Temperature vs Density phase diagram.

        Background shows pressure contours (log scale) with the saturation dome
        and coolant path overlaid.

        Args:
            cooling_result: CoolingResult with density array populated
            fluid_name: CoolProp fluid identifier
            title: Optional figure title override

        Returns:
            Plotly Figure
        """
        try:
            import CoolProp.CoolProp as CP  # noqa: F401
        except ImportError:
            raise RuntimeError("CoolProp is required for phase diagram generation")

        T_crit, P_crit, rho_crit, T_trip = self._get_critical(fluid_name)

        # --- Background: pressure on T-ρ grid ---
        T_min, T_max = 200.0, 360.0
        rho_min, rho_max = 1.0, 1200.0
        n_T, n_rho = 120, 120

        rho_axis = np.linspace(rho_min, rho_max, n_rho)
        T_axis = np.linspace(T_min, T_max, n_T)
        RHO, T_MESH = np.meshgrid(rho_axis, T_axis)

        P_grid = self._calc_property_grid(
            "P", "T", T_MESH.ravel(), "D", RHO.ravel(), fluid_name
        ).reshape(T_MESH.shape)

        P_grid_bar = P_grid / 1e5
        P_grid_bar = np.where(P_grid_bar > 0, P_grid_bar, np.nan)

        # Use log scale for pressure
        log_P = np.log10(np.clip(P_grid_bar, 0.1, 500))
        log_ticks = [1, 10, 50, 72.45, 100, 200]
        log_tick_vals = [np.log10(v) for v in log_ticks]

        fig = go.Figure()

        # Pressure heatmap background
        fig.add_trace(go.Heatmap(
            x=rho_axis,
            y=T_axis,
            z=log_P,
            colorscale=self.theme.pressure_colorscale,
            showscale=True,
            opacity=0.75,
            hovertemplate="ρ=%{x:.0f} kg/m³<br>T=%{y:.1f} K<br>P=%{customdata:.1f} bar<extra></extra>",
            customdata=P_grid_bar,
            colorbar=dict(
                title="Pressure [bar]",
                tickvals=log_tick_vals,
                ticktext=[str(v) for v in log_ticks],
                thickness=15,
                len=0.85,
            ),
            name="",
            showlegend=False,
        ))

        # Saturation dome
        T_sat = np.linspace(T_trip, T_crit - 0.2, 250)
        rho_liq, rho_vap = self._saturation_envelope(T_sat, fluid_name)

        dome_color = "rgba(255,255,255,0.9)" if _is_dark(self.theme) else "rgba(50,50,50,0.9)"

        fig.add_trace(go.Scatter(
            x=rho_liq, y=T_sat,
            mode='lines', name="Sat. Liquid",
            line=dict(color=dome_color, width=2),
            hovertemplate="ρ_liq=%{x:.1f} kg/m³<br>T=%{y:.1f} K<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=rho_vap, y=T_sat,
            mode='lines', name="Sat. Vapor",
            line=dict(color=dome_color, width=2, dash='dot'),
            hovertemplate="ρ_vap=%{x:.1f} kg/m³<br>T=%{y:.1f} K<extra></extra>",
        ))

        # Critical point
        fig.add_trace(go.Scatter(
            x=[rho_crit], y=[T_crit],
            mode='markers+text',
            name="Critical Point",
            marker=dict(color='yellow', size=10, symbol='star'),
            text=["Critical"],
            textposition="top right",
            textfont=dict(color='yellow', size=11),
        ))

        # Coolant path
        self._add_coolant_path_t_rho(fig, cooling_result)

        # Region annotations
        ann_color = "rgba(255,255,255,0.7)" if _is_dark(self.theme) else "rgba(40,40,40,0.7)"
        _regions = [
            (250, 345, "Supercritical"),
            (1000, 215, "Liquid"),
            (30, 215, "Vapor"),
            (500, 290, "Two-Phase"),
        ]
        for rho_ann, T_ann, label in _regions:
            fig.add_annotation(
                x=rho_ann, y=T_ann,
                text=label,
                showarrow=False,
                font=dict(size=12, color=ann_color),
            )

        fig.update_layout(
            title=dict(text=title or "N₂O State Diagram: Temperature vs Density", x=0.5),
            xaxis=dict(title="Density ρ [kg/m³]", range=[rho_min, rho_max]),
            yaxis=dict(title="Temperature T [K]", range=[T_min, T_max]),
            height=600,
            showlegend=True,
        )
        self.theme.apply_to_figure(fig)
        return fig

    def create_p_t_figure(
        self,
        results: Union["CoolingResult", List["CoolingResult"], Dict[str, "CoolingResult"]],
        fluid_name: str = "NitrousOxide",
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create Pressure vs Temperature phase diagram.

        Background shows density contours with saturation line, critical point,
        and one or more coolant paths.

        Args:
            results: Single CoolingResult, list, or labeled dict
            fluid_name: CoolProp fluid identifier
            title: Optional figure title override

        Returns:
            Plotly Figure
        """
        try:
            import CoolProp.CoolProp as CP  # noqa: F401
        except ImportError:
            raise RuntimeError("CoolProp is required for phase diagram generation")

        # Normalize input to dict
        results_dict = _normalize_results(results)

        T_crit, P_crit, rho_crit, T_trip = self._get_critical(fluid_name)

        # Determine plot bounds from data
        all_T, all_P = [], []
        for r in results_dict.values():
            all_T.extend(r.T_coolant.tolist())
            all_P.extend((r.P_coolant / 1e5).tolist())

        T_min = max(T_trip, min(all_T) - 5) if all_T else 200.0
        T_max = max(360.0, max(all_T) + 10) if all_T else 360.0
        P_min_bar = max(0.5, min(all_P) - 2) if all_P else 1.0
        P_max_bar = max(100.0, max(all_P) + 5) if all_P else 100.0

        n_T, n_P = 100, 100
        T_axis = np.linspace(T_min, T_max, n_T)
        P_axis_bar = np.linspace(P_min_bar, P_max_bar, n_P)
        T_MESH, P_MESH = np.meshgrid(T_axis, P_axis_bar)

        rho_grid = self._calc_property_grid(
            "D", "T", T_MESH.ravel(), "P", (P_MESH * 1e5).ravel(), fluid_name
        ).reshape(T_MESH.shape)

        fig = go.Figure()

        # Density heatmap background
        fig.add_trace(go.Heatmap(
            x=T_axis,
            y=P_axis_bar,
            z=rho_grid,
            colorscale=self.theme.temperature_colorscale,
            reversescale=True,
            showscale=True,
            opacity=0.65,
            hovertemplate="T=%{x:.1f} K<br>P=%{y:.1f} bar<br>ρ=%{z:.1f} kg/m³<extra></extra>",
            colorbar=dict(
                title="Density [kg/m³]",
                thickness=15,
                len=0.85,
            ),
            name="",
            showlegend=False,
        ))

        # Saturation line
        T_sat = np.linspace(T_trip, T_crit - 0.2, 250)
        P_sat = self._saturation_pressure(T_sat, fluid_name) / 1e5

        sat_color = "rgba(255,255,255,0.9)" if _is_dark(self.theme) else "rgba(30,30,30,0.9)"
        fig.add_trace(go.Scatter(
            x=T_sat, y=P_sat,
            mode='lines', name="Saturation Line",
            line=dict(color=sat_color, width=2.5),
            hovertemplate="T_sat=%{x:.1f} K<br>P_sat=%{y:.2f} bar<extra></extra>",
        ))

        # Critical point
        fig.add_trace(go.Scatter(
            x=[T_crit], y=[P_crit / 1e5],
            mode='markers+text',
            name="Critical Point",
            marker=dict(color='yellow', size=10, symbol='star'),
            text=["Critical"],
            textposition="top right",
            textfont=dict(color='yellow', size=11),
        ))

        # Reference lines at critical T and P
        ref_color = "rgba(255,255,255,0.2)" if _is_dark(self.theme) else "rgba(0,0,0,0.15)"
        fig.add_vline(x=T_crit, line_dash="dot", line_color=ref_color, line_width=1)
        fig.add_hline(y=P_crit / 1e5, line_dash="dot", line_color=ref_color, line_width=1)

        # Coolant paths
        colors = self.theme.get_color_sequence()
        for idx, (label, r) in enumerate(results_dict.items()):
            color = colors[idx % len(colors)]
            T_path = r.T_coolant
            P_path = r.P_coolant / 1e5

            fig.add_trace(go.Scatter(
                x=T_path, y=P_path,
                mode='lines', name=label,
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{label}</b><br>T=%{{x:.1f}} K<br>P=%{{y:.2f}} bar<extra></extra>",
            ))
            # Inlet marker
            fig.add_trace(go.Scatter(
                x=[T_path[0]], y=[P_path[0]],
                mode='markers', name=f"{label} — Inlet",
                marker=dict(color=color, size=9, symbol='circle'),
                showlegend=(idx == 0),
                legendgroup="inlet_markers",
                hovertemplate="<b>Inlet</b><br>T=%{x:.1f} K<br>P=%{y:.2f} bar<extra></extra>",
            ))
            # Outlet marker
            fig.add_trace(go.Scatter(
                x=[T_path[-1]], y=[P_path[-1]],
                mode='markers', name=f"{label} — Outlet",
                marker=dict(color=color, size=9, symbol='x'),
                showlegend=(idx == 0),
                legendgroup="outlet_markers",
                hovertemplate="<b>Outlet</b><br>T=%{x:.1f} K<br>P=%{y:.2f} bar<extra></extra>",
            ))

        fig.update_layout(
            title=dict(text=title or "N₂O State Diagram: Pressure vs Temperature", x=0.5),
            xaxis=dict(title="Temperature T [K]", range=[T_min, T_max]),
            yaxis=dict(title="Pressure P [bar]", range=[P_min_bar, P_max_bar]),
            height=600,
            showlegend=True,
        )
        self.theme.apply_to_figure(fig)
        return fig

    def to_html(
        self,
        fig: go.Figure,
        include_plotlyjs: str = 'cdn',
        full_html: bool = False,
    ) -> str:
        """Export figure to embeddable HTML."""
        return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=full_html)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_critical(self, fluid_name: str):
        """Return (T_crit, P_crit, rho_crit, T_triple) using CoolProp."""
        try:
            import CoolProp.CoolProp as CP
            T_crit = CP.PropsSI("T_critical", fluid_name)
            P_crit = CP.PropsSI("p_critical", fluid_name)
            rho_crit = CP.PropsSI("rhocrit", fluid_name)
            try:
                T_triple = CP.PropsSI("T_triple", fluid_name)
            except Exception:
                T_triple = self._T_TRIPLE
        except Exception:
            T_crit = self._T_CRIT
            P_crit = self._P_CRIT
            rho_crit = self._RHO_CRIT
            T_triple = self._T_TRIPLE
        return T_crit, P_crit, rho_crit, T_triple

    @staticmethod
    def _calc_property_grid(
        out_prop: str,
        in1_prop: str,
        in1_vals: np.ndarray,
        in2_prop: str,
        in2_vals: np.ndarray,
        fluid_name: str,
    ) -> np.ndarray:
        """Vectorized CoolProp call with NaN fallback for invalid states."""
        import CoolProp.CoolProp as CP
        try:
            result = CP.PropsSI(out_prop, in1_prop, in1_vals, in2_prop, in2_vals, fluid_name)
            result = np.where(np.isfinite(result), result, np.nan)
            return result
        except Exception:
            pass
        # Scalar fallback loop
        result = np.empty(len(in1_vals))
        for k in range(len(in1_vals)):
            try:
                result[k] = CP.PropsSI(out_prop, in1_prop, in1_vals[k], in2_prop, in2_vals[k], fluid_name)
            except Exception:
                result[k] = np.nan
        return result

    def _saturation_envelope(self, T_arr, fluid_name):
        """Return (rho_liq, rho_vap) saturation densities for T_arr."""
        import CoolProp.CoolProp as CP
        rho_liq = np.array([
            CP.PropsSI("D", "T", t, "Q", 0, fluid_name) for t in T_arr
        ])
        rho_vap = np.array([
            CP.PropsSI("D", "T", t, "Q", 1, fluid_name) for t in T_arr
        ])
        return rho_liq, rho_vap

    def _saturation_pressure(self, T_arr, fluid_name):
        """Return saturation pressure [Pa] for T_arr."""
        import CoolProp.CoolProp as CP
        return np.array([
            CP.PropsSI("P", "T", t, "Q", 0, fluid_name) for t in T_arr
        ])

    def _add_coolant_path_t_rho(self, fig: go.Figure, result: "CoolingResult"):
        """Add the coolant T-ρ path trace to fig."""
        if result.density is None:
            return  # No density data; skip

        T_path = result.T_coolant
        rho_path = result.density
        color = self.theme.secondary

        fig.add_trace(go.Scatter(
            x=rho_path, y=T_path,
            mode='lines+markers',
            name="Coolant Path",
            line=dict(color=color, width=2.5),
            marker=dict(color=color, size=4),
            hovertemplate="ρ=%{x:.1f} kg/m³<br>T=%{y:.1f} K<extra></extra>",
        ))
        # Inlet/outlet markers
        for idx, label, symbol in [(0, "Inlet", "circle"), (-1, "Outlet", "x")]:
            fig.add_trace(go.Scatter(
                x=[rho_path[idx]], y=[T_path[idx]],
                mode='markers+text',
                name=label,
                marker=dict(color=color, size=12, symbol=symbol),
                text=[label],
                textposition="top center",
                textfont=dict(color=color, size=11),
                showlegend=False,
            ))


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _is_dark(theme: PlotTheme) -> bool:
    """Heuristic: theme has a dark background if paper_background is dark."""
    try:
        bg = theme.paper_background.strip().lstrip("#")
        if len(bg) == 6:
            r, g, b = int(bg[0:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
            return (r + g + b) / 3 < 100
    except Exception:
        pass
    return False


def _normalize_results(
    results: Union["CoolingResult", List["CoolingResult"], Dict[str, "CoolingResult"]]
) -> Dict[str, "CoolingResult"]:
    """Normalize input to {label: CoolingResult} dict."""
    from resa.core.results import CoolingResult
    if isinstance(results, CoolingResult):
        return {"Design Point": results}
    if isinstance(results, list):
        return {f"Run {i + 1}": r for i, r in enumerate(results)}
    if isinstance(results, dict):
        return results
    raise TypeError(f"Expected CoolingResult, list, or dict; got {type(results)}")
