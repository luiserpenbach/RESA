"""
RESA - Rocket Engine Sizing & Analysis
Core module providing interfaces, results, and configuration classes.
"""
from .config import EngineConfig, AnalysisPreset
from .results import (
    EngineDesignResult,
    CombustionResult,
    CoolingResult,
    NozzleGeometry,
    CoolingChannelGeometry,
    ThrottlePoint,
    ThrottleCurve
)
from .interfaces import (
    FluidState,
    BaseSolver,
    CombustionSolverInterface,
    CoolingSolverInterface,
    FlowComponent
)
from .exceptions import (
    RESAError,
    ConfigurationError,
    ConvergenceError,
    ThermodynamicError,
    GeometryError,
    CoolingError
)

__all__ = [
    # Config
    'EngineConfig',
    'AnalysisPreset',
    
    # Results
    'EngineDesignResult',
    'CombustionResult',
    'CoolingResult',
    'NozzleGeometry',
    'CoolingChannelGeometry',
    'ThrottlePoint',
    'ThrottleCurve',
    
    # Interfaces
    'FluidState',
    'BaseSolver',
    'CombustionSolverInterface',
    'CoolingSolverInterface',
    'FlowComponent',
    
    # Exceptions
    'RESAError',
    'ConfigurationError',
    'ConvergenceError',
    'ThermodynamicError',
    'GeometryError',
    'CoolingError',
]
