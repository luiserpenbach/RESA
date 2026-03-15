"""
Feed system analysis API routes.
"""
from __future__ import annotations

import asyncio
import logging
from functools import partial

from fastapi import APIRouter, HTTPException, Query

from api.models.feed_system_models import (
    FeedSystemConfigRequest,
    FeedSystemResponse,
)
from api.services.session_manager import session_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feed-system", tags=["feed-system"])


def _to_config(req: FeedSystemConfigRequest):
    """Convert Pydantic request model to FeedSystemConfig dataclass."""
    from resa.core.module_configs import FeedSystemConfig

    return FeedSystemConfig(**req.model_dump())


def _run_feed_system(session, config):
    """Run feed system analysis (synchronous)."""
    return session.run_feed_system(config)


@router.post("/analyze", response_model=FeedSystemResponse)
async def analyze_feed_system(
    req: FeedSystemConfigRequest,
    session_id: str = Query(..., description="Design session ID"),
):
    """
    Run full feed system analysis including pressure budget, line losses,
    and pump/turbine sizing (if pump-fed).
    """
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    if session.engine_result is None:
        raise HTTPException(
            status_code=400,
            detail="Engine design must be run before feed system analysis",
        )

    config = _to_config(req)
    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(
            None, partial(_run_feed_system, session, config)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Feed system analysis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Collect warnings
    warnings: list[str] = []
    if result.npsh_available_m > 0 and result.npsh_margin_m < 0:
        warnings.append(
            f"NPSH margin is negative ({result.npsh_margin_m:.2f} m) — cavitation risk"
        )
    elif result.npsh_available_m > 0 and result.npsh_margin_m < 1.0:
        warnings.append(
            f"NPSH margin ({result.npsh_margin_m:.2f} m) is low"
        )

    # required_feed_pressure = total pressure the feed system must deliver
    required_feed = (
        result.pump_discharge_pressure_bar
        if result.pump_discharge_pressure_bar > 0
        else result.tank_pressure_bar
    )

    return FeedSystemResponse(
        feed_type=result.feed_type,
        cycle_type=result.cycle_type,
        required_feed_pressure_bar=required_feed,
        injector_dp_bar=result.injector_dp_bar,
        cooling_dp_bar=result.cooling_dp_bar,
        line_losses_ox_bar=result.line_losses_ox_bar,
        line_losses_fuel_bar=result.line_losses_fuel_bar,
        pump_power_ox_w=result.pump_power_ox_w,
        pump_power_fuel_w=result.pump_power_fuel_w,
        pump_head_ox_m=result.pump_head_ox_m,
        pump_head_fuel_m=result.pump_head_fuel_m,
        npsh_available_m=result.npsh_available_m,
        turbine_power_w=result.turbine_power_w,
        power_balance_margin_pct=result.power_balance_margin * 100.0,
        warnings=warnings,
    )
