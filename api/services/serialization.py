"""
Serialization helpers: convert RESA result objects to JSON-safe dicts.
Handles numpy arrays and Plotly figure serialization.
"""
from __future__ import annotations

import math
import logging
from typing import Any

logger = logging.getLogger(__name__)

R_UNIVERSAL = 8314.462618  # J/(kmol·K)


def _safe_figure(plotter_callable) -> str | None:
    """Call a plotter factory, return fig.to_json() or None on failure."""
    try:
        fig = plotter_callable()
        return fig.to_json()
    except Exception as exc:
        logger.warning("Figure generation failed: %s", exc)
        return None


def _station_props(mach: float, Tc: float, Pc_bar: float, gamma: float,
                   mw: float) -> dict:
    """
    Compute isentropic thermodynamic state at a given Mach number.

    Args:
        mach: Local Mach number
        Tc: Stagnation (chamber) temperature [K]
        Pc_bar: Stagnation pressure [bar]
        gamma: Specific heat ratio
        mw: Molecular weight [g/mol = kg/kmol]

    Returns:
        Dict with T_k, P_bar, rho, V_ms, mach
    """
    T = Tc / (1.0 + (gamma - 1.0) / 2.0 * mach ** 2)
    P = Pc_bar * (T / Tc) ** (gamma / (gamma - 1.0))
    R_spec = R_UNIVERSAL / mw  # J/(kg·K)
    rho = (P * 1e5) / (R_spec * T)
    a = math.sqrt(gamma * R_spec * T)  # speed of sound [m/s]
    V = mach * a
    return {
        "T_k": round(T, 1),
        "P_bar": round(P, 4),
        "rho": round(rho, 3),
        "V_ms": round(V, 1),
        "mach": round(mach, 4),
    }


def serialize_design_result(result, config) -> dict[str, Any]:
    """
    Convert EngineDesignResult to a JSON-safe dict with embedded Plotly figures.

    Args:
        result: EngineDesignResult instance
        config: EngineConfig instance (used by some plotters)

    Returns:
        Dict suitable for constructing EngineDesignResponse
    """
    from resa.visualization.engine_plots import EnginePrimaryDashboardPlotter
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

    # Station thermodynamics (chamber ≈ M=0.05, throat M=1, exit M_exit)
    if result.combustion:
        comb = result.combustion
        Tc = comb.T_combustion
        Pc = result.pc_bar
        gamma = comb.gamma
        mw = comb.mw
        Me = comb.mach_exit

        # Chamber: approximate inlet Mach ~0.05 (subsonic, near-stagnation)
        out["station_chamber"] = _station_props(0.05, Tc, Pc, gamma, mw)
        # Throat: M = 1.0 exactly (choked)
        out["station_throat"] = _station_props(1.0, Tc, Pc, gamma, mw)
        # Exit: from CEA equilibrium
        out["station_exit"] = _station_props(Me, Tc, Pc, gamma, mw)

    # Cooling summary
    if result.cooling:
        out["max_wall_temp"] = float(result.cooling.max_wall_temp)
        out["max_heat_flux"] = float(result.cooling.max_heat_flux)
        out["pressure_drop_bar"] = float(result.cooling.pressure_drop)
        out["outlet_temp_k"] = float(result.cooling.outlet_temp)

    # Nozzle geometry — contour arrays + complete dimensions
    if result.nozzle_geometry is not None:
        geo = result.nozzle_geometry
        x_mm = geo.x_full * 1000.0
        y_mm = geo.y_full * 1000.0
        out["contour_x_mm"] = x_mm.tolist()
        out["contour_y_mm"] = y_mm.tolist()

        # Chamber diameter
        out["dc_mm"] = round(geo.chamber_radius * 2000.0, 2)

        # Contraction ratio
        if geo.throat_radius > 0 and geo.chamber_radius > 0:
            out["contraction_ratio"] = round(
                (geo.chamber_radius / geo.throat_radius) ** 2, 3
            )

        # Exit angle in degrees
        if geo.theta_exit:
            out["theta_exit_deg"] = round(math.degrees(geo.theta_exit), 2)

        # Section lengths from geometry arrays
        if geo.x_chamber is not None and len(geo.x_chamber) > 1:
            out["L_chamber_mm"] = round(
                float((geo.x_chamber[-1] - geo.x_chamber[0]) * 1000.0), 1
            )
        if geo.x_convergent is not None and len(geo.x_convergent) > 1:
            out["L_convergent_mm"] = round(
                float((geo.x_convergent[-1] - geo.x_convergent[0]) * 1000.0), 1
            )
        if geo.x_divergent is not None and len(geo.x_divergent) > 1:
            out["L_divergent_mm"] = round(
                float((geo.x_divergent[-1] - geo.x_divergent[0]) * 1000.0), 1
            )

    # Primary dashboard figure (works without cooling)
    out["figure_dashboard"] = _safe_figure(
        lambda: EnginePrimaryDashboardPlotter(theme).create_figure(result, config)
    )

    return out
