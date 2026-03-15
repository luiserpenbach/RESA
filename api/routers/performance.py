"""
Performance map API routes (altitude curves, throttle maps).
"""
from __future__ import annotations

import asyncio
import logging
from functools import partial

from fastapi import APIRouter, HTTPException, Query

from api.models.performance_models import (
    AltitudePerformanceResponse,
    PerformanceFullResponse,
    PerformanceMapConfigRequest,
    ThrottleMapResponse,
)
from api.services.session_manager import session_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/performance", tags=["performance"])


def _to_config(req: PerformanceMapConfigRequest):
    """Convert Pydantic request to PerformanceMapConfig dataclass."""
    from resa.core.module_configs import PerformanceMapConfig

    return PerformanceMapConfig(
        altitude_range_m=(req.altitude_range_min_m, req.altitude_range_max_m),
        altitude_points=req.altitude_points,
        throttle_range=(req.throttle_range_min, req.throttle_range_max),
        throttle_points=req.throttle_points,
    )


def _run_full(session, config):
    """Run full performance map via DesignSession (synchronous)."""
    return session.run_performance_map(config)


@router.post("/full", response_model=PerformanceFullResponse)
async def full_performance(
    session_id: str = Query(..., description="Design session ID"),
    req: PerformanceMapConfigRequest | None = None,
):
    """Compute both altitude and throttle performance maps in one call."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    if session.engine_result is None:
        raise HTTPException(
            status_code=400,
            detail="Engine design must be run before performance analysis",
        )

    config = _to_config(req or PerformanceMapConfigRequest())
    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(None, partial(_run_full, session, config))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Full performance analysis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Build altitude response
    altitude_resp = None
    if result.altitudes_m is not None:
        sep_alt = None
        if result.cf_vs_alt is not None:
            # Find separation altitude where Cf first goes negative (if any)
            import numpy as np

            neg = np.where(result.cf_vs_alt < 0)[0]
            if len(neg) > 0:
                sep_alt = float(result.altitudes_m[neg[0]])

        altitude_resp = AltitudePerformanceResponse(
            altitudes_m=result.altitudes_m.tolist(),
            thrust_n=result.thrust_vs_alt.tolist(),
            isp_s=result.isp_vs_alt.tolist(),
            cf=result.cf_vs_alt.tolist() if result.cf_vs_alt is not None else [],
            separation_altitude_m=sep_alt,
        )

    # Build throttle response
    throttle_resp = None
    if result.throttle_pcts is not None:
        throttle_resp = ThrottleMapResponse(
            throttle_pcts=(result.throttle_pcts * 100).tolist(),
            pc_bar=result.pc_vs_throttle.tolist(),
            thrust_n=result.thrust_vs_throttle.tolist(),
            isp_s=result.isp_vs_throttle.tolist(),
        )

    return PerformanceFullResponse(
        altitude=altitude_resp,
        throttle=throttle_resp,
    )
