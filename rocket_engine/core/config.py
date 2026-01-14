"""
DEPRECATED: This module is deprecated. Please use resa.core.config instead.

This file now imports from resa.core.config to avoid code duplication.
"""
import warnings

warnings.warn(
    "rocket_engine.core.config is deprecated and will be removed in a future version. "
    "Please use resa.core.config instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new location
from resa.core.config import *

__all__ = ['EngineConfig', 'AnalysisPreset', 'PROPELLANT_ALIASES', 'MATERIAL_CONDUCTIVITY', 'ValidationResult']
