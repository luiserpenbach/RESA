"""
RESA FastAPI application.
Run with: uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import engine, config_io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RESA API starting up")
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
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"
app.include_router(engine.router, prefix=API_PREFIX)
app.include_router(config_io.router, prefix=API_PREFIX)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "RESA API", "version": "2.0.0"}
