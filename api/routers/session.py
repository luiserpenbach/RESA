"""
Session management API routes.
"""
from __future__ import annotations

import asyncio
import logging
from functools import partial

from fastapi import APIRouter, HTTPException

from api.models.engine_models import EngineConfigRequest
from api.models.session_models import (
    SessionCreateResponse,
    SessionStatusResponse,
)
from api.services.serialization import serialize_design_result
from api.services.session_manager import session_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/session", tags=["session"])


def _inflate_config(req: EngineConfigRequest):
    from resa.core.config import EngineConfig

    return EngineConfig(**req.model_dump())


def _create_and_run(config, with_cooling: bool):
    """Create session and run initial engine design (synchronous)."""
    session_id = session_manager.create_session(config)
    session = session_manager.get_session(session_id)
    session.run_engine_design(with_cooling=with_cooling)
    return session_id, session


@router.post("/create", response_model=SessionCreateResponse)
async def create_session(
    req: EngineConfigRequest,
    with_cooling: bool = False,
):
    """Create a new design session and run the initial engine design."""
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

    loop = asyncio.get_event_loop()
    try:
        session_id, session = await loop.run_in_executor(
            None, partial(_create_and_run, config, with_cooling)
        )
    except Exception as exc:
        logger.exception("Session creation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Serialize the engine design result so the frontend has it immediately
    engine_data = None
    if session.engine_result is not None:
        try:
            engine_data = await loop.run_in_executor(
                None,
                partial(serialize_design_result, session.engine_result, config),
            )
        except Exception as exc:
            logger.warning("Engine result serialization failed: %s", exc)

    return SessionCreateResponse(
        session_id=session_id,
        module_status=session.get_module_status(),
        engine_result=engine_data,
    )


@router.get("/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """Get the current status of all modules in a session."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    return SessionStatusResponse(
        session_id=session_id,
        module_status=session.get_module_status(),
    )


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a design session."""
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}
