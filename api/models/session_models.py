"""
Pydantic models for session management API.
"""
from __future__ import annotations

from pydantic import BaseModel


class SessionCreateResponse(BaseModel):
    session_id: str
    module_status: dict[str, str]


class SessionStatusResponse(BaseModel):
    session_id: str
    module_status: dict[str, str]
