"""
RESA FastAPI application.

Single-server mode: FastAPI serves the React build from web/dist at /
API routes are under /api/v1

Run with: uvicorn api.main:app --reload --port 8000
Then open:  http://localhost:8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routers import config_io, cooling, engine, session, structural

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the compiled React app
WEB_DIST = Path(__file__).parent.parent / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RESA API starting up")
    if WEB_DIST.exists():
        logger.info("Serving React frontend from %s", WEB_DIST)
    else:
        logger.warning("web/dist not found — run 'cd web && npm run build' to build the frontend")
    yield
    logger.info("RESA API shutting down")


app = FastAPI(
    title="RESA API",
    description="Rocket Engine Sizing & Analysis REST API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes (registered first so they take priority) ─────────────────────
API_PREFIX = "/api/v1"
app.include_router(engine.router, prefix=API_PREFIX)
app.include_router(config_io.router, prefix=API_PREFIX)
app.include_router(session.router, prefix=API_PREFIX)
app.include_router(cooling.router, prefix=API_PREFIX)
app.include_router(structural.router, prefix=API_PREFIX)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "RESA API", "version": "2.0.0"}


# ── React static files ───────────────────────────────────────────────────────
# Mount /assets (JS/CSS chunks) from the build output
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Serve index.html for all non-API routes (React Router client-side routing)."""
        # Serve actual files that exist (favicon, etc.)
        requested = WEB_DIST / full_path
        if requested.is_file():
            return FileResponse(requested)
        # Fall back to index.html for all React Router routes
        return FileResponse(WEB_DIST / "index.html")
