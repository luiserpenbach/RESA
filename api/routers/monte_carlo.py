"""
Monte Carlo uncertainty analysis API routes.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from functools import partial
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from api.models.monte_carlo_models import (
    MonteCarloConfigRequest,
    MonteCarloResponse,
    OutputStatistics,
)
from api.services.session_manager import session_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/monte-carlo", tags=["monte-carlo"])

# Supported engine config parameters that can be perturbed
SUPPORTED_PARAMS = {
    "pc_bar", "mr", "thrust_n", "eff_combustion",
    "expansion_ratio", "contraction_ratio",
}


def _extract_output(result: Any, name: str) -> float:
    """Extract a named output from EngineDesignResult using dotted path."""
    parts = name.split(".")
    obj = result
    for part in parts:
        obj = getattr(obj, part, None)
        if obj is None:
            raise KeyError(f"Output '{name}' not found in result")
    return float(obj)


def _run_mc(session, req: MonteCarloConfigRequest):
    from resa.analysis.monte_carlo import MonteCarloAnalysis
    from resa.core.engine import Engine

    base_config = session.config

    # Validate requested parameters
    for p in req.parameters:
        if p.name not in SUPPORTED_PARAMS:
            raise ValueError(
                f"Parameter '{p.name}' is not supported for perturbation. "
                f"Supported: {sorted(SUPPORTED_PARAMS)}"
            )

    mc = MonteCarloAnalysis(seed=42)
    for p in req.parameters:
        mc.add_parameter(
            name=p.name,
            nominal=p.nominal,
            distribution=p.distribution,
            std_dev=p.std_dev,
            min_val=p.min_val,
            max_val=p.max_val,
            mode=p.mode,
        )

    if not req.parameters:
        # No parameters defined → just run with defaults for pc_bar and mr
        mc.add_parameter("pc_bar", base_config.pc_bar, "normal", std_dev=base_config.pc_bar * 0.03)
        mc.add_parameter("mr", base_config.mr, "normal", std_dev=base_config.mr * 0.03)

    def engine_func(**kwargs):
        cfg = replace(base_config, **kwargs)
        engine = Engine(cfg)
        result = engine.design(with_cooling=False)
        outputs = {}
        for name in req.output_names:
            try:
                outputs[name] = _extract_output(result, name)
            except (KeyError, AttributeError, TypeError):
                outputs[name] = float("nan")
        return outputs

    mc_result = mc.run(
        n_samples=req.n_samples,
        engine_func=engine_func,
        output_names=req.output_names,
    )
    return mc_result


@router.post("/run", response_model=MonteCarloResponse)
async def run_monte_carlo(
    session_id: str = Query(..., description="Design session ID"),
    req: MonteCarloConfigRequest | None = None,
):
    """Run Monte Carlo uncertainty analysis on engine parameters."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if session.engine_result is None:
        raise HTTPException(
            status_code=400,
            detail="Engine design must be run before Monte Carlo analysis",
        )

    req = req or MonteCarloConfigRequest()
    loop = asyncio.get_event_loop()

    try:
        mc_result = await loop.run_in_executor(None, partial(_run_mc, session, req))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Monte Carlo analysis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    stats_out = {}
    for name in mc_result.output_names:
        s = mc_result.statistics.get(name, {})
        stats_out[name] = OutputStatistics(
            mean=s.get("mean", 0.0),
            std=s.get("std", 0.0),
            p5=s.get("P5", 0.0),
            p50=s.get("P50", 0.0),
            p95=s.get("P95", 0.0),
        )

    sensitivity_out: dict[str, dict[str, float]] = {}
    for out_name, sens in mc_result.sensitivity.items():
        pearson = sens.get("pearson", {})
        sensitivity_out[out_name] = {k: float(v) for k, v in pearson.items()}

    samples_out = {
        name: [float(v) for v in mc_result.output_samples[name]]
        for name in mc_result.output_names
        if name in mc_result.output_samples
    }

    return MonteCarloResponse(
        n_samples=mc_result.n_samples,
        n_failed=mc_result.failed_samples,
        elapsed_s=mc_result.elapsed_time,
        statistics=stats_out,
        sensitivity=sensitivity_out,
        output_samples=samples_out,
    )
