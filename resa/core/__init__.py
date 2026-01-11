"""
Core domain models and interfaces for RESA.

This module provides the foundational classes for:
- Engine configuration and results
- Solver interfaces for physics calculations
- Geometry generation contracts
- Fluid property providers
- Visualization and reporting interfaces
"""

from resa.core.config import EngineConfig, ValidationResult, AnalysisPreset
from resa.core.results import (
    CombustionResult,
    NozzleGeometry,
    CoolingChannelGeometry,
    CoolingResult,
    EngineDesignResult,
    InjectorGeometryResult,
    InjectorPerformanceResult,
    ThrottlePoint,
    ThrottleCurve,
)
from resa.core.interfaces import (
    Solver,
    CombustionSolver,
    CoolingSolver,
    GeometryGenerator,
    FluidProvider,
    FluidState,
    Plotter,
    Viewer3D,
    ReportGenerator,
    AnalysisModule,
    VersionControl,
    DesignVersion,
    OutputManager,
    OutputConfig,
    MonteCarloEngine,
    UncertaintyParameter,
    MonteCarloResult,
    Optimizer,
    OptimizationConstraint,
    OptimizationResult,
)
from resa.core.exceptions import (
    RESAError,
    ConfigurationError,
    ConvergenceError,
    ThermodynamicError,
    GeometryError,
    CombustionError,
    CoolingError,
    FlowModelError,
    MaterialLimitError,
    RESAWarning,
    PerformanceWarning,
    StabilityWarning,
    ThermalWarning,
)

__all__ = [
    # Configuration
    "EngineConfig",
    "ValidationResult",
    "AnalysisPreset",
    # Results
    "CombustionResult",
    "NozzleGeometry",
    "CoolingChannelGeometry",
    "CoolingResult",
    "EngineDesignResult",
    "InjectorGeometryResult",
    "InjectorPerformanceResult",
    "ThrottlePoint",
    "ThrottleCurve",
    # Interfaces
    "Solver",
    "CombustionSolver",
    "CoolingSolver",
    "GeometryGenerator",
    "FluidProvider",
    "FluidState",
    "Plotter",
    "Viewer3D",
    "ReportGenerator",
    "AnalysisModule",
    "VersionControl",
    "DesignVersion",
    "OutputManager",
    "OutputConfig",
    "MonteCarloEngine",
    "UncertaintyParameter",
    "MonteCarloResult",
    "Optimizer",
    "OptimizationConstraint",
    "OptimizationResult",
    # Exceptions
    "RESAError",
    "ConfigurationError",
    "ConvergenceError",
    "ThermodynamicError",
    "GeometryError",
    "CombustionError",
    "CoolingError",
    "FlowModelError",
    "MaterialLimitError",
    "RESAWarning",
    "PerformanceWarning",
    "StabilityWarning",
    "ThermalWarning",
]
