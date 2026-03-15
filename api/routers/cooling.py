"""
Cooling design and analysis API routes.
"""
from __future__ import annotations

import asyncio
import logging
from functools import partial

import numpy as np
from fastapi import APIRouter, HTTPException, Query

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
    aspect = np.where(geom.channel_width > 0, geom.channel_height / geom.channel_width, 0)

    return CoolingChannelResponse(
        num_channels=geom.num_channels,
        channel_type=req.channel_type,
        height_profile=req.height_profile,
        min_channel_width_mm=float(np.min(width_mm)),
        max_channel_width_mm=float(np.max(width_mm)),
        min_aspect_ratio=float(np.min(aspect)),
        max_aspect_ratio=float(np.max(aspect)),
        x_mm=(geom.x * 1e3).tolist(),
        channel_width_mm=width_mm.tolist(),
        channel_height_mm=height_mm.tolist(),
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
        from resa.visualization.engine_plots import EngineDashboardPlotter
        from resa.visualization.themes import DarkTheme

        engine_result = session.engine_result
        if engine_result is not None:
            plotter = EngineDashboardPlotter(theme=DarkTheme())
            fig = plotter.create_figure(engine_result)
            figure_json = fig.to_json()
    except Exception:
        logger.warning("Could not generate thermal figure", exc_info=True)

    nozzle = session.engine_result.nozzle_geometry if session.engine_result else None
    x_mm = (nozzle.x_full * 1e3).tolist() if nozzle is not None else []

    warnings = []
    if result.max_wall_temp > 900:
        warnings.append(
            f"Max wall temperature {result.max_wall_temp:.0f} K exceeds 900 K"
        )

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
    )
