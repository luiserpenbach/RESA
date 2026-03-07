"""
Engine design API routes.
"""
from __future__ import annotations

import asyncio
import logging
from functools import partial

from fastapi import APIRouter, HTTPException, Query

from api.models.engine_models import (
    EngineConfigRequest,
    EngineDesignResponse,
    ValidationResponse,
)
from api.services.serialization import serialize_design_result

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/engine", tags=["engine"])


def _inflate_config(req: EngineConfigRequest):
    """Convert Pydantic request model to EngineConfig dataclass."""
    from resa.core.config import EngineConfig

    return EngineConfig(**req.model_dump())


def _run_design(config, with_cooling: bool):
    """Synchronous design run — executed in thread pool."""
    from resa.core.engine import Engine

    engine = Engine(config)
    return engine.design(with_cooling=with_cooling)


@router.post("/validate", response_model=ValidationResponse)
async def validate_config(req: EngineConfigRequest) -> ValidationResponse:
    """Validate an engine configuration. Returns errors and warnings."""
    try:
        config = _inflate_config(req)
        result = config.validate()
        return ValidationResponse(
            is_valid=result.is_valid,
            errors=list(result.errors),
            warnings=list(result.warnings),
        )
    except Exception as exc:
        logger.exception("Config validation error")
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/design", response_model=EngineDesignResponse)
async def design_engine(
    req: EngineConfigRequest,
    with_cooling: bool = Query(default=False, description="Run full cooling analysis"),
) -> EngineDesignResponse:
    """
    Run engine design. Returns performance metrics and serialized Plotly figures.

    Set with_cooling=false (default) for fast iterations without thermal analysis.
    Set with_cooling=true for full regenerative cooling solve (slower).
    """
    try:
        config = _inflate_config(req)
        validation = config.validate()
        if not validation.is_valid:
            raise HTTPException(
                status_code=422,
                detail={"errors": validation.errors, "warnings": validation.warnings},
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Run solver in thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, partial(_run_design, config, with_cooling)
        )
    except Exception as exc:
        logger.exception("Engine design failed")
        raise HTTPException(status_code=500, detail=f"Design failed: {exc}") from exc

    # Serialize (also in thread pool — figure generation can be slow)
    try:
        data = await loop.run_in_executor(
            None, partial(serialize_design_result, result, config)
        )
    except Exception as exc:
        logger.exception("Serialization failed")
        raise HTTPException(status_code=500, detail=f"Serialization failed: {exc}") from exc

    return EngineDesignResponse(**data)
