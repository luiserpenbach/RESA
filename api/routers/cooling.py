"""
Cooling design and analysis API routes.
"""
from __future__ import annotations

import asyncio
import logging
from functools import partial

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from api.models.cooling_models import (
    CoolingAnalysisResponse,
    CoolingChannelConfigRequest,
    CoolingChannelResponse,
)
from api.services.session_manager import session_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cooling", tags=["cooling"])


def _to_config(req: CoolingChannelConfigRequest):
    from resa.core.module_configs import CoolingChannelConfig

    return CoolingChannelConfig(**req.model_dump())


def _run_channels(session, config):
    """Generate cooling channels (synchronous)."""
    geom = session.run_cooling_channels(config)
    return geom


def _run_analysis(session):
    """Run cooling thermal analysis (synchronous)."""
    return session.run_cooling_analysis()


@router.post("/channels", response_model=CoolingChannelResponse)
async def design_channels(
    req: CoolingChannelConfigRequest,
    session_id: str = Query(..., description="Design session ID"),
):
    """Generate cooling channel geometry on the current nozzle contour."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    config = _to_config(req)
    loop = asyncio.get_event_loop()

    try:
        geom = await loop.run_in_executor(None, partial(_run_channels, session, config))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Channel design failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    width_mm = geom.channel_width * 1e3
    height_mm = geom.channel_height * 1e3
    rib_mm = geom.rib_width * 1e3
    inner_r_mm = geom.y * 1e3
    aspect = np.where(geom.channel_width > 0, geom.channel_height / geom.channel_width, 0)
    wall_thickness_mm_val = float(np.mean(geom.wall_thickness) * 1e3)

    # Generate 3D channel figure
    figure_3d = None
    try:
        from resa.visualization.engine_3d import Engine3DViewer
        from resa.visualization.themes import DarkTheme

        viewer = Engine3DViewer(theme=DarkTheme(), dark_mode=True)
        fig3d = viewer.render_channels(geom, n_channels_to_show=4, resolution=40)
        figure_3d = fig3d.to_json()
    except Exception:
        logger.warning("Could not generate 3D channel figure", exc_info=True)

    return CoolingChannelResponse(
        num_channels=geom.num_channels,
        channel_type=req.channel_type,
        height_profile=req.height_profile,
        min_channel_width_mm=float(np.min(width_mm)),
        max_channel_width_mm=float(np.max(width_mm)),
        min_aspect_ratio=float(np.min(aspect)),
        max_aspect_ratio=float(np.max(aspect)),
        wall_thickness_mm_val=wall_thickness_mm_val,
        x_mm=(geom.x * 1e3).tolist(),
        channel_width_mm=width_mm.tolist(),
        channel_height_mm=height_mm.tolist(),
        rib_width_mm=rib_mm.tolist(),
        inner_radius_mm=inner_r_mm.tolist(),
        figure_3d=figure_3d,
    )


@router.post("/analyze", response_model=CoolingAnalysisResponse)
async def analyze_cooling(
    session_id: str = Query(..., description="Design session ID"),
):
    """Run thermal-hydraulic cooling analysis on current channel geometry."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(None, partial(_run_analysis, session))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Cooling analysis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Generate thermal dashboard figure
    figure_json = None
    try:
        import dataclasses

        from resa.visualization.engine_plots import EngineDashboardPlotter
        from resa.visualization.themes import DarkTheme

        engine_result = session.engine_result
        if engine_result is not None:
            # Merge the freshly computed cooling result into engine_result so the
            # plotter has access to thermal arrays (engine_result.cooling is None
            # at this point because cooling runs as a separate session step).
            engine_result_with_cooling = dataclasses.replace(engine_result, cooling=result)
            plotter = EngineDashboardPlotter(theme=DarkTheme())
            fig = plotter.create_figure(engine_result_with_cooling)
            figure_json = fig.to_json()
    except Exception:
        logger.warning("Could not generate thermal figure", exc_info=True)

    # Generate N2O phase diagram figures (only when density data is available)
    figure_t_rho = None
    figure_p_t = None
    is_n2o = result.density is not None
    if is_n2o:
        try:
            from resa.visualization.cooling_plots import CoolingPhaseDiagramPlotter
            from resa.visualization.themes import DarkTheme

            phase_plotter = CoolingPhaseDiagramPlotter(theme=DarkTheme())
            fig_t_rho = phase_plotter.create_t_rho_figure(result)
            figure_t_rho = fig_t_rho.to_json()
            fig_p_t = phase_plotter.create_p_t_figure(result)
            figure_p_t = fig_p_t.to_json()
        except Exception:
            logger.warning("Could not generate N2O phase diagram figures", exc_info=True)

    # Generate 3D thermal channel figure
    figure_3d_thermal = None
    channel_geom = session.get_result("cooling_channels")
    if channel_geom is not None:
        try:
            from resa.visualization.engine_3d import Engine3DViewer
            from resa.visualization.themes import DarkTheme

            viewer = Engine3DViewer(theme=DarkTheme(), dark_mode=True)
            fig_3d_th = viewer.render_single_channel_thermal(
                channel_geom,
                temperature_data=result.T_wall_hot,
                colorbar_title='T_wall_hot [K]',
            )
            figure_3d_thermal = fig_3d_th.to_json()
        except Exception:
            logger.warning("Could not generate 3D thermal channel figure", exc_info=True)

    nozzle = session.engine_result.nozzle_geometry if session.engine_result else None
    x_mm = (nozzle.x_full * 1e3).tolist() if nozzle is not None else []

    warnings = []
    if result.max_wall_temp > 900:
        warnings.append(
            f"Max wall temperature {result.max_wall_temp:.0f} K exceeds 900 K"
        )
    if is_n2o and result.min_chf_margin is not None and result.min_chf_margin > 0.5:
        warnings.append(
            f"CHF margin exceeded at worst station (q/q_CHF = {result.min_chf_margin:.2f}). "
            "Risk of wall burnout."
        )
    if is_n2o and result.flow_regime is not None and "post_chf" in result.flow_regime:
        warnings.append("POST-CHF condition detected — film boiling likely. Redesign required.")

    return CoolingAnalysisResponse(
        max_wall_temp_k=result.max_wall_temp,
        max_heat_flux_mw_m2=result.max_heat_flux / 1e6,
        pressure_drop_bar=result.pressure_drop,
        outlet_temp_k=result.outlet_temp,
        figure_thermal=figure_json,
        x_mm=x_mm,
        t_wall_hot_k=result.T_wall_hot.tolist(),
        t_wall_cold_k=result.T_wall_cold.tolist(),
        t_coolant_k=result.T_coolant.tolist(),
        q_flux_mw_m2=(result.q_flux / 1e6).tolist(),
        coolant_velocity_m_s=result.velocity.tolist(),
        coolant_pressure_bar=(result.P_coolant / 1e5).tolist(),
        warnings=warnings,
        # N2O-specific
        is_n2o_analysis=is_n2o,
        min_chf_margin=result.min_chf_margin,
        max_quality=result.max_quality,
        chf_margin=result.chf_margin.tolist() if result.chf_margin is not None else None,
        vapor_quality=result.quality.tolist() if result.quality is not None else None,
        flow_regime=result.flow_regime,
        h_conv_kw_m2k=(result.h_conv / 1e3).tolist() if result.h_conv is not None else None,
        density_kg_m3=result.density.tolist() if result.density is not None else None,
        figure_t_rho=figure_t_rho,
        figure_p_t=figure_p_t,
        figure_3d_thermal=figure_3d_thermal,
    )


@router.get("/cross_section")
async def get_cross_section(
    session_id: str = Query(..., description="Design session ID"),
    station_idx: int = Query(0, description="Axial station index"),
):
    """Return a Plotly cross-section figure for a specific axial station."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    channel_geom = session.get_result("cooling_channels")
    if channel_geom is None:
        raise HTTPException(
            status_code=400,
            detail="No channel geometry available. Run /cooling/channels first.",
        )

    # Clamp station index to valid range
    n_stations = len(channel_geom.x)
    station_idx = max(0, min(station_idx, n_stations - 1))

    try:
        from resa.visualization.engine_plots import CrossSectionPlotter
        from resa.visualization.themes import DarkTheme

        plotter = CrossSectionPlotter(theme=DarkTheme())
        fig = plotter.create_figure(channel_geom, station_idx=station_idx)
        return JSONResponse(content={"figure": fig.to_json()})
    except Exception as exc:
        logger.exception("Cross-section figure generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
