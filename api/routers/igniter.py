"""
Torch igniter design API routes.
"""
from __future__ import annotations

import asyncio
import logging
from functools import partial

from fastapi import APIRouter, HTTPException

from api.models.igniter_models import (
    IgniterConfigRequest,
    IgniterCombustionResponse,
    IgniterGeometryResponse,
    IgniterInjectorResponse,
    IgniterMassFlowsResponse,
    IgniterPerformanceResponse,
    IgniterResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/igniter", tags=["igniter"])


def _run_design(req: IgniterConfigRequest):
    from resa.addons.igniter.config import IgniterConfig
    from resa.addons.igniter.designer import IgniterDesigner

    config = IgniterConfig(
        chamber_pressure=req.chamber_pressure_pa,
        mixture_ratio=req.mixture_ratio,
        total_mass_flow=req.total_mass_flow_kg_s,
        ethanol_feed_pressure=req.ethanol_feed_pressure_pa,
        n2o_feed_pressure=req.n2o_feed_pressure_pa,
        ethanol_feed_temperature=req.ethanol_feed_temperature_k,
        n2o_feed_temperature=req.n2o_feed_temperature_k,
        l_star=req.l_star,
        expansion_ratio=req.expansion_ratio,
        nozzle_type=req.nozzle_type,
        n2o_orifice_count=req.n2o_orifice_count,
        ethanol_orifice_count=req.ethanol_orifice_count,
        discharge_coefficient=req.discharge_coefficient,
    )
    designer = IgniterDesigner()
    return designer.design(config)


@router.post("/design", response_model=IgniterResponse)
async def design_igniter(req: IgniterConfigRequest | None = None):
    """Size a torch igniter from operating conditions."""
    req = req or IgniterConfigRequest()
    loop = asyncio.get_event_loop()
    try:
        r = await loop.run_in_executor(None, partial(_run_design, req))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Igniter design failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IgniterResponse(
        combustion=IgniterCombustionResponse(
            flame_temperature_k=r.flame_temperature,
            c_star_m_s=r.c_star,
            gamma=r.gamma,
            molecular_weight=r.molecular_weight,
            heat_power_kw=r.heat_power_output / 1000.0,
        ),
        geometry=IgniterGeometryResponse(
            chamber_diameter_mm=r.chamber_diameter * 1000,
            chamber_length_mm=r.chamber_length * 1000,
            chamber_volume_cm3=r.chamber_volume * 1e6,
            throat_diameter_mm=r.throat_diameter * 1000,
            exit_diameter_mm=r.exit_diameter * 1000,
            nozzle_length_mm=r.nozzle_length * 1000,
        ),
        injector=IgniterInjectorResponse(
            n2o_orifice_diameter_mm=r.n2o_orifice_diameter * 1000,
            ethanol_orifice_diameter_mm=r.ethanol_orifice_diameter * 1000,
            n2o_injection_velocity_m_s=r.n2o_injection_velocity,
            ethanol_injection_velocity_m_s=r.ethanol_injection_velocity,
            n2o_pressure_drop_bar=r.n2o_pressure_drop / 1e5,
            ethanol_pressure_drop_bar=r.ethanol_pressure_drop / 1e5,
        ),
        performance=IgniterPerformanceResponse(
            isp_theoretical_s=r.isp_theoretical,
            thrust_n=r.thrust,
        ),
        mass_flows=IgniterMassFlowsResponse(
            total_kg_s=r.total_mass_flow,
            oxidizer_kg_s=r.oxidizer_mass_flow,
            fuel_kg_s=r.fuel_mass_flow,
            mixture_ratio=r.mixture_ratio,
        ),
    )
