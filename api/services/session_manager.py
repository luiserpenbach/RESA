"""
In-memory session manager for RESA design sessions.

Maps session_id (UUID) -> DesignSession with TTL expiry.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Dict, Optional

from resa.core.config import EngineConfig
from resa.core.session import DesignSession

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 30 * 60  # 30 minutes


class SessionEntry:
    __slots__ = ("session", "last_access")

    def __init__(self, session: DesignSession):
        self.session = session
        self.last_access = time.monotonic()

    def touch(self):
        self.last_access = time.monotonic()


class SessionManager:
    """Thread-safe in-memory session store."""

    def __init__(self, ttl: int = SESSION_TTL_SECONDS):
        self._sessions: Dict[str, SessionEntry] = {}
        self._lock = threading.Lock()
        self._ttl = ttl

    def create_session(self, config: EngineConfig) -> str:
        """Create a new design session and return its ID."""
        session_id = str(uuid.uuid4())
        session = DesignSession(config)
        with self._lock:
            self._sessions[session_id] = SessionEntry(session)
        logger.info("Created session %s", session_id)
        self._cleanup_expired()
        return session_id

    def get_session(self, session_id: str) -> Optional[DesignSession]:
        """Get session by ID, updating last-access time."""
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            entry.touch()
            return entry.session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    def _cleanup_expired(self):
        """Remove sessions older than TTL."""
        now = time.monotonic()
        with self._lock:
            expired = [
                sid
                for sid, entry in self._sessions.items()
                if now - entry.last_access > self._ttl
            ]
            for sid in expired:
                del self._sessions[sid]
                logger.info("Expired session %s", sid)


# Global singleton
session_manager = SessionManager()
