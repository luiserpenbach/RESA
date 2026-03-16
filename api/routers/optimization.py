"""
Design optimization API routes.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from functools import partial

from fastapi import APIRouter, HTTPException, Query

from api.models.optimization_models import (
    OptimizationConfigRequest,
    OptimizationResponse,
)
from api.services.session_manager import session_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/optimization", tags=["optimization"])

# Engine config fields that can be optimized
SUPPORTED_VARIABLES = {"pc_bar", "mr", "thrust_n", "expansion_ratio", "eff_combustion"}

# Available outputs from EngineDesignResult
SUPPORTED_OUTPUTS = {"isp_vac", "isp_sea", "thrust_vac", "thrust_sea", "massflow_total"}


def _extract_output(result, name: str) -> float:
    parts = name.split(".")
    obj = result
    for part in parts:
        obj = getattr(obj, part, None)
        if obj is None:
            raise KeyError(f"Output '{name}' not found")
    return float(obj)


def _run_optimization(session, req: OptimizationConfigRequest):
    from resa.analysis.optimization import ThrottleOptimizer
    from resa.core.engine import Engine

    base_config = session.config

    for v in req.variables:
        if v.name not in SUPPORTED_VARIABLES:
            raise ValueError(
                f"Variable '{v.name}' not supported. Supported: {sorted(SUPPORTED_VARIABLES)}"
            )

    optimizer = ThrottleOptimizer(method=req.algorithm)

    # If no variables specified, default to pc_bar and mr
    if not req.variables:
        optimizer.add_variable("pc_bar", base_config.pc_bar * 0.5, base_config.pc_bar * 2.0, base_config.pc_bar)
        optimizer.add_variable("mr", max(1.0, base_config.mr * 0.7), base_config.mr * 1.5, base_config.mr)
    else:
        for v in req.variables:
            optimizer.add_variable(v.name, v.min_val, v.max_val, v.initial)

    for c in req.constraints:
        optimizer.add_constraint(c.output_name, c.output_name, c.type, c.limit)

    optimizer.set_objective(req.objective, minimize=req.minimize)

    all_output_names = {req.objective} | {c.output_name for c in req.constraints}

    def eval_func(variables: dict) -> dict:
        cfg = replace(base_config, **variables)
        engine = Engine(cfg)
        result = engine.design(with_cooling=False)
        outputs = {}
        for name in all_output_names:
            try:
                outputs[name] = _extract_output(result, name)
            except (KeyError, AttributeError, TypeError):
                outputs[name] = 0.0
        return outputs

    opt_result = optimizer.optimize(eval_func, max_iterations=req.max_iterations)
    return opt_result


@router.post("/run", response_model=OptimizationResponse)
async def run_optimization(
    session_id: str = Query(..., description="Design session ID"),
    req: OptimizationConfigRequest | None = None,
):
    """Optimize engine design variables to maximize/minimize an objective."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if session.engine_result is None:
        raise HTTPException(
            status_code=400,
            detail="Engine design must be run before optimization",
        )

    req = req or OptimizationConfigRequest()
    loop = asyncio.get_event_loop()

    try:
        opt_result = await loop.run_in_executor(None, partial(_run_optimization, session, req))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Optimization failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    history_iter = opt_result.history.iterations
    history_obj = opt_result.history.objective_values

    return OptimizationResponse(
        optimal_variables={k: float(v) for k, v in opt_result.optimal_variables.items()},
        optimal_outputs={k: float(v) for k, v in opt_result.optimal_outputs.items()},
        objective_value=float(opt_result.optimal_objective),
        n_evaluations=opt_result.n_evaluations,
        converged=opt_result.success,
        message=opt_result.message,
        history_iterations=list(history_iter),
        history_objective=[float(v) for v in history_obj],
    )
