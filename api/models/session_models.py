"""
Pydantic models for session management API.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SessionCreateResponse(BaseModel):
    session_id: str
    module_status: dict[str, str]
    engine_result: dict[str, Any] | None = None


class SessionStatusResponse(BaseModel):
    session_id: str
    module_status: dict[str, str]
