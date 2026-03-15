"""
Pydantic models for nozzle contour API.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ContourConfigRequest(BaseModel):
    """Config for contour generation - supplements the session's engine design."""

    bell_fraction: Optional[float] = None  # override session's bell_fraction
    theta_exit: Optional[float] = None
    resolution: int = 200  # number of contour points
    wall_thickness_mm: float = 1.0  # for 3D/export


class ContourResponse(BaseModel):
    """Response with 2D contour data."""

    # Dimensions
    throat_diameter_mm: float
    exit_diameter_mm: float
    total_length_mm: float
    expansion_ratio: float

    # Contour arrays (mm)
    x_mm: list[float]
    y_mm: list[float]

    # Section breakdown
    x_chamber_mm: list[float]
    y_chamber_mm: list[float]
    x_convergent_mm: list[float]
    y_convergent_mm: list[float]
    x_divergent_mm: list[float]
    y_divergent_mm: list[float]

    # Plotly figure JSON
    figure_contour: Optional[str] = None
    figure_3d: Optional[str] = None


class ContourExportRequest(BaseModel):
    format: str = "csv"  # csv, stl, json
    wall_thickness_mm: float = 1.0
