"""
RESA - Rocket Engine Sizing & Analysis
======================================

A comprehensive Python toolkit for liquid rocket engine preliminary design,
including combustion analysis, regenerative cooling, and injector design.

Modules:
--------
- core: Configuration, results, and interface definitions
- physics: Pure physics calculations (combustion, heat transfer, fluid flow)
- geometry: Nozzle and cooling channel geometry generation
- solvers: Integrated analysis solvers
- components: Feed system and injector models
- analysis: Performance analysis and visualization
- ui: Streamlit-based graphical interface

Quick Start:
------------
>>> from rocket_engine.core import EngineConfig
>>> config = EngineConfig(
...     engine_name="MyEngine",
...     fuel="Ethanol90",
...     oxidizer="N2O",
...     thrust_n=2000,
...     pc_bar=25,
...     mr=4.0
... )
>>> 
>>> # Run with Streamlit UI:
>>> # streamlit run rocket_engine/ui/app.py

Author: RESA Development Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "RESA Development Team"

import warnings
warnings.warn(
    "The 'rocket_engine' package is deprecated and will be removed in a future version. "
    "Please migrate to the 'resa' package instead.",
    DeprecationWarning,
    stacklevel=2
)

# Convenience imports
from .core import (
    EngineConfig,
    EngineDesignResult,
    CombustionResult,
    AnalysisPreset
)

__all__ = [
    'EngineConfig',
    'EngineDesignResult', 
    'CombustionResult',
    'AnalysisPreset',
    '__version__',
]
