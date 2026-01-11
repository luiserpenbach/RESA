"""Core domain models and interfaces for RESA."""

from resa.core.interfaces import (
    Solver,
    GeometryGenerator,
    FluidProvider,
    Plotter,
    ReportGenerator,
)

__all__ = [
    "Solver",
    "GeometryGenerator",
    "FluidProvider",
    "Plotter",
    "ReportGenerator",
]
