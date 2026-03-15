"""
Structural / wall thickness analysis API routes.
"""
from __future__ import annotations

import asyncio
import logging
from functools import partial

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from api.models.structural_models import (
    MaterialListResponse,
    MaterialPropertiesResponse,
    WallThicknessConfigRequest,
    WallThicknessResponse,
)
from api.services.session_manager import session_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/structural", tags=["structural"])


def _run_wall_thickness(session, config):
    """Run wall thickness analysis (synchronous)."""
    from resa.core.module_configs import WallThicknessConfig

    wt_config = WallThicknessConfig(**config.model_dump())
    return session.run_wall_thickness(wt_config)


@router.post("/wall-thickness", response_model=WallThicknessResponse)
async def analyze_wall_thickness(
    req: WallThicknessConfigRequest,
    session_id: str = Query(..., description="Design session ID"),
):
    """Compute wall thickness requirements and stress distribution."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(
            None, partial(_run_wall_thickness, session, req)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Wall thickness analysis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    x_mm = (result.x * 1e3).tolist()
    sf = result.safety_factor
    min_sf_idx = int(np.argmin(sf))

    mat = result.material
    mat_response = MaterialPropertiesResponse(
        name=mat.name,
        density_kg_m3=mat.density_kg_m3,
        yield_strength_mpa=mat.yield_strength_pa / 1e6,
        thermal_conductivity_w_mk=mat.thermal_conductivity_w_mk,
        max_service_temp_k=mat.max_service_temp_k,
    )

    warnings = []
    if float(np.min(sf)) < 1.0:
        warnings.append(
            f"Safety factor below 1.0 at x={x_mm[min_sf_idx]:.1f} mm — wall is overstressed"
        )
    elif float(np.min(sf)) < 1.5:
        warnings.append(
            f"Safety factor {float(np.min(sf)):.2f} at x={x_mm[min_sf_idx]:.1f} mm is marginal"
        )

    return WallThicknessResponse(
        min_safety_factor=float(np.min(sf)),
        critical_station_x_mm=float(x_mm[min_sf_idx]),
        max_hoop_stress_mpa=float(np.max(result.hoop_stress) / 1e6),
        max_thermal_stress_mpa=float(np.max(result.thermal_stress) / 1e6),
        max_von_mises_mpa=float(np.max(result.von_mises_stress) / 1e6),
        material=mat_response,
        x_mm=x_mm,
        min_thickness_pressure_mm=(result.min_thickness_pressure * 1e3).tolist(),
        min_thickness_thermal_mm=(result.min_thickness_thermal * 1e3).tolist(),
        actual_thickness_mm=(result.actual_thickness * 1e3).tolist(),
        safety_factor=sf.tolist(),
        hoop_stress_mpa=(result.hoop_stress / 1e6).tolist(),
        thermal_stress_mpa=(result.thermal_stress / 1e6).tolist(),
        von_mises_mpa=(result.von_mises_stress / 1e6).tolist(),
        warnings=warnings,
    )


@router.get("/materials", response_model=MaterialListResponse)
async def list_materials():
    """List all available materials."""
    from resa.core.materials import list_materials

    return MaterialListResponse(materials=list_materials())
