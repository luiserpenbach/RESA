"""
Nozzle contour generation and export API routes.
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from functools import partial
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.models.contour_models import ContourConfigRequest, ContourResponse
from api.services.session_manager import session_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/contour", tags=["contour"])


def _get_session_or_raise(session_id: str):
    """Retrieve a session or raise 404."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return session


def _require_engine_result(session):
    """Return engine_result or raise 400 if not available."""
    result = session.engine_result
    if result is None:
        raise HTTPException(
            status_code=400,
            detail="Engine design has not been run. Run engine design first.",
        )
    return result


def _generate_contour(session_id: str, req: ContourConfigRequest) -> dict:
    """Synchronous contour generation - runs in thread pool."""
    import numpy as np
    from resa.geometry.nozzle import NozzleGenerator

    session = _get_session_or_raise(session_id)
    engine_result = _require_engine_result(session)

    nozzle_geom = engine_result.nozzle_geometry
    if nozzle_geom is None:
        raise HTTPException(
            status_code=400,
            detail="No nozzle geometry available in engine result.",
        )

    # If overrides are provided, re-generate the nozzle contour
    bell_fraction = req.bell_fraction
    theta_exit = req.theta_exit

    if bell_fraction is not None or theta_exit is not None:
        generator = NozzleGenerator()
        gen_kwargs = {
            "throat_radius": nozzle_geom.throat_radius,
            "expansion_ratio": (nozzle_geom.exit_radius / nozzle_geom.throat_radius) ** 2,
            "bell_fraction": bell_fraction if bell_fraction is not None else 0.8,
        }
        if nozzle_geom.chamber_radius > 0:
            gen_kwargs["R_chamber"] = nozzle_geom.chamber_radius
        nozzle_geom = generator.generate(**gen_kwargs)

    # Convert arrays from meters to mm
    x_full_raw = nozzle_geom.x_full * 1000
    y_full_raw = nozzle_geom.y_full * 1000

    # Resample to requested resolution
    n_orig = len(x_full_raw)
    n_target = max(10, req.resolution)
    if n_target != n_orig:
        t_orig = np.linspace(0, 1, n_orig)
        t_new = np.linspace(0, 1, n_target)
        x_full_mm = np.interp(t_new, t_orig, x_full_raw).tolist()
        y_full_mm = np.interp(t_new, t_orig, y_full_raw).tolist()
    else:
        x_full_mm = x_full_raw.tolist()
        y_full_mm = y_full_raw.tolist()

    # Section arrays (keep original resolution for plot accuracy)
    x_chamber_mm = (
        (nozzle_geom.x_chamber * 1000).tolist()
        if nozzle_geom.x_chamber is not None
        else []
    )
    y_chamber_mm = (
        (nozzle_geom.y_chamber * 1000).tolist()
        if nozzle_geom.y_chamber is not None
        else []
    )
    x_convergent_mm = (
        (nozzle_geom.x_convergent * 1000).tolist()
        if nozzle_geom.x_convergent is not None
        else []
    )
    y_convergent_mm = (
        (nozzle_geom.y_convergent * 1000).tolist()
        if nozzle_geom.y_convergent is not None
        else []
    )
    x_divergent_mm = (
        (nozzle_geom.x_divergent * 1000).tolist()
        if nozzle_geom.x_divergent is not None
        else []
    )
    y_divergent_mm = (
        (nozzle_geom.y_divergent * 1000).tolist()
        if nozzle_geom.y_divergent is not None
        else []
    )

    throat_diameter_mm = nozzle_geom.throat_radius * 2000
    exit_diameter_mm = nozzle_geom.exit_radius * 2000
    total_length_mm = nozzle_geom.total_length
    expansion_ratio = (nozzle_geom.exit_radius / nozzle_geom.throat_radius) ** 2

    # Generate 2D contour Plotly figure
    figure_contour = _build_contour_figure(
        x_full_mm,
        y_full_mm,
        x_chamber_mm,
        y_chamber_mm,
        x_convergent_mm,
        y_convergent_mm,
        x_divergent_mm,
        y_divergent_mm,
    )

    # Generate 3D figure if possible
    figure_3d = _build_3d_figure(nozzle_geom)

    return {
        "throat_diameter_mm": throat_diameter_mm,
        "exit_diameter_mm": exit_diameter_mm,
        "total_length_mm": total_length_mm,
        "expansion_ratio": expansion_ratio,
        "x_mm": x_full_mm,
        "y_mm": y_full_mm,
        "x_chamber_mm": x_chamber_mm,
        "y_chamber_mm": y_chamber_mm,
        "x_convergent_mm": x_convergent_mm,
        "y_convergent_mm": y_convergent_mm,
        "x_divergent_mm": x_divergent_mm,
        "y_divergent_mm": y_divergent_mm,
        "figure_contour": figure_contour,
        "figure_3d": figure_3d,
    }


def _build_contour_figure(
    x_full_mm,
    y_full_mm,
    x_chamber_mm,
    y_chamber_mm,
    x_convergent_mm,
    y_convergent_mm,
    x_divergent_mm,
    y_divergent_mm,
) -> Optional[str]:
    """Build a 2D Plotly contour figure and return as JSON string."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        if x_chamber_mm:
            fig.add_trace(
                go.Scatter(
                    x=x_chamber_mm,
                    y=y_chamber_mm,
                    mode="lines",
                    name="Chamber",
                    line=dict(color="#1f77b4", width=2),
                )
            )
            # Mirror
            fig.add_trace(
                go.Scatter(
                    x=x_chamber_mm,
                    y=[-v for v in y_chamber_mm],
                    mode="lines",
                    name="Chamber",
                    line=dict(color="#1f77b4", width=2),
                    showlegend=False,
                )
            )

        if x_convergent_mm:
            fig.add_trace(
                go.Scatter(
                    x=x_convergent_mm,
                    y=y_convergent_mm,
                    mode="lines",
                    name="Convergent",
                    line=dict(color="#ff7f0e", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_convergent_mm,
                    y=[-v for v in y_convergent_mm],
                    mode="lines",
                    name="Convergent",
                    line=dict(color="#ff7f0e", width=2),
                    showlegend=False,
                )
            )

        if x_divergent_mm:
            fig.add_trace(
                go.Scatter(
                    x=x_divergent_mm,
                    y=y_divergent_mm,
                    mode="lines",
                    name="Divergent",
                    line=dict(color="#2ca02c", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_divergent_mm,
                    y=[-v for v in y_divergent_mm],
                    mode="lines",
                    name="Divergent",
                    line=dict(color="#2ca02c", width=2),
                    showlegend=False,
                )
            )

        fig.update_layout(
            title="Nozzle Contour",
            xaxis_title="Axial Position [mm]",
            yaxis_title="Radial Position [mm]",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            template="plotly_white",
        )

        return fig.to_json()
    except Exception:
        logger.warning("Failed to generate 2D contour figure", exc_info=True)
        return None


def _build_3d_figure(nozzle_geom) -> Optional[str]:
    """Build a 3D Plotly figure using Engine3DViewer if available."""
    try:
        from resa.visualization.engine_3d import Engine3DViewer

        viewer = Engine3DViewer()
        fig = viewer.render_nozzle(nozzle_geom)
        return fig.to_json()
    except Exception:
        logger.warning("Failed to generate 3D figure", exc_info=True)
        return None


def _export_contour_data(session_id: str, fmt: str) -> tuple:
    """Synchronous export - runs in thread pool. Returns (content, media_type, filename)."""
    session = _get_session_or_raise(session_id)
    engine_result = _require_engine_result(session)

    nozzle_geom = engine_result.nozzle_geometry
    if nozzle_geom is None:
        raise HTTPException(
            status_code=400,
            detail="No nozzle geometry available in engine result.",
        )

    x_mm = (nozzle_geom.x_full * 1000).tolist()
    y_mm = (nozzle_geom.y_full * 1000).tolist()

    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["x_mm", "y_mm"])
        for x, y in zip(x_mm, y_mm):
            writer.writerow([f"{x:.6f}", f"{y:.6f}"])
        content = buf.getvalue()
        return content, "text/csv", "nozzle_contour.csv"

    elif fmt == "json":
        data = {
            "throat_diameter_mm": nozzle_geom.throat_radius * 2000,
            "exit_diameter_mm": nozzle_geom.exit_radius * 2000,
            "total_length_mm": nozzle_geom.total_length,
            "expansion_ratio": (nozzle_geom.exit_radius / nozzle_geom.throat_radius) ** 2,
            "x_mm": x_mm,
            "y_mm": y_mm,
        }
        content = json.dumps(data, indent=2)
        return content, "application/json", "nozzle_contour.json"

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported export format: '{fmt}'. Use 'csv' or 'json'.",
        )


@router.post("/generate", response_model=ContourResponse)
async def generate_contour(
    req: ContourConfigRequest,
    session_id: str = Query(..., description="Design session ID"),
) -> ContourResponse:
    """
    Generate or refine the nozzle contour for a design session.

    Uses the session's existing engine result nozzle geometry. If ``bell_fraction``
    or ``theta_exit`` overrides are provided, the nozzle is re-generated with the
    new parameters while keeping the same throat/exit sizing.
    """
    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(
            None, partial(_generate_contour, session_id, req)
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Contour generation failed")
        raise HTTPException(status_code=500, detail=f"Contour generation failed: {exc}") from exc

    return ContourResponse(**data)


@router.get("/export")
async def export_contour(
    session_id: str = Query(..., description="Design session ID"),
    format: str = Query(default="csv", description="Export format: csv or json"),
):
    """
    Export the nozzle contour data as a downloadable file.

    Supported formats:
    - **csv**: Two-column CSV with x_mm, y_mm
    - **json**: JSON object with geometry metadata and coordinate arrays
    """
    loop = asyncio.get_event_loop()
    try:
        content, media_type, filename = await loop.run_in_executor(
            None, partial(_export_contour_data, session_id, format)
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Contour export failed")
        raise HTTPException(status_code=500, detail=f"Contour export failed: {exc}") from exc

    return StreamingResponse(
        io.StringIO(content),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
