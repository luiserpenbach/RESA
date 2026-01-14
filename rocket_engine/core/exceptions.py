"""
DEPRECATED: This module is deprecated. Please use resa.core.exceptions instead.

This file now imports from resa.core.exceptions to avoid code duplication.
"""
import warnings

warnings.warn(
    "rocket_engine.core.exceptions is deprecated and will be removed in a future version. "
    "Please use resa.core.exceptions instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new location
from resa.core.exceptions import *

__all__ = [
    'RESAError',
    'ConfigurationError',
    'ConvergenceError',
    'ThermodynamicError',
    'GeometryError',
    'CombustionError',
    'CoolingError',
    'FlowModelError',
    'MaterialLimitError',
    'RESAWarning',
    'PerformanceWarning',
    'StabilityWarning',
    'ThermalWarning',
]
