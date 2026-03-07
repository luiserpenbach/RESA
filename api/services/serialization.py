"""
Serialization helpers: convert RESA result objects to JSON-safe dicts.
Handles numpy arrays and Plotly figure serialization.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _safe_figure(plotter_callable) -> str | None:
    """Call a plotter factory, return fig.to_json() or None on failure."""
    try:
        fig = plotter_callable()
        return fig.to_json()
    except Exception as exc:
        logger.warning("Figure generation failed: %s", exc)
        return None


def serialize_design_result(result, config) -> dict[str, Any]:
    """
    Convert EngineDesignResult to a JSON-safe dict with embedded Plotly figures.

    Args:
        result: EngineDesignResult instance
        config: EngineConfig instance (used by some plotters)

    Returns:
        Dict suitable for constructing EngineDesignResponse
    """
    from resa.visualization.engine_plots import (
        EngineDashboardPlotter,
        NozzleContourPlotter,
        GasDynamicsPlotter,
    )
    from resa.visualization.engine_3d import Engine3DViewer
    from resa.visualization.themes import DarkTheme

    theme = DarkTheme()

    out: dict[str, Any] = {
        "timestamp": result.timestamp.isoformat() if result.timestamp else "",
        "run_type": result.run_type,
        "pc_bar": result.pc_bar,
        "mr": result.mr,
        "isp_vac": result.isp_vac,
        "isp_sea": result.isp_sea,
        "thrust_vac": result.thrust_vac,
        "thrust_sea": result.thrust_sea,
        "massflow_total": result.massflow_total,
        "massflow_ox": result.massflow_ox,
        "massflow_fuel": result.massflow_fuel,
        "dt_mm": result.dt_mm,
        "de_mm": result.de_mm,
        "length_mm": result.length_mm,
        "expansion_ratio": result.expansion_ratio,
        "warnings": list(result.warnings),
    }

    # Combustion sub-result
    if result.combustion:
        out["combustion"] = {
            "pc_bar": result.combustion.pc_bar,
            "mr": result.combustion.mr,
            "cstar": result.combustion.cstar,
            "isp_vac": result.combustion.isp_vac,
            "isp_opt": result.combustion.isp_opt,
            "T_combustion": result.combustion.T_combustion,
            "gamma": result.combustion.gamma,
            "mw": result.combustion.mw,
            "mach_exit": result.combustion.mach_exit,
        }

    # Cooling summary
    if result.cooling:
        out["max_wall_temp"] = float(result.cooling.max_wall_temp)
        out["max_heat_flux"] = float(result.cooling.max_heat_flux)
        out["pressure_drop_bar"] = float(result.cooling.pressure_drop)
        out["outlet_temp_k"] = float(result.cooling.outlet_temp)

    # Nozzle contour arrays
    if result.nozzle_geometry is not None:
        geo = result.nozzle_geometry
        out["contour_x_mm"] = (geo.x_full * 1000.0).tolist()
        out["contour_y_mm"] = (geo.y_full * 1000.0).tolist()

    # Plotly figures
    out["figure_dashboard"] = _safe_figure(
        lambda: EngineDashboardPlotter(theme).create_figure(result)
    )
    out["figure_contour"] = _safe_figure(
        lambda: NozzleContourPlotter(theme).create_figure(result)
    )
    out["figure_gas_dynamics"] = _safe_figure(
        lambda: GasDynamicsPlotter(theme).create_figure(result)
    )
    if result.nozzle_geometry is not None and result.channel_geometry is not None:
        out["figure_3d"] = _safe_figure(
            lambda: Engine3DViewer(theme).render_full_engine(
                result.nozzle_geometry, result.channel_geometry
            )
        )
    elif result.nozzle_geometry is not None:
        out["figure_3d"] = _safe_figure(
            lambda: Engine3DViewer(theme).render_nozzle(result.nozzle_geometry)
        )
    else:
        out["figure_3d"] = None

    return out
