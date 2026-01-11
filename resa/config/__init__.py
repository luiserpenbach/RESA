"""
Configuration management for RESA.

Provides:
- EngineConfig dataclass for engine parameters
- YAML/JSON configuration loading
- Validation utilities
"""

from resa.config.engine_config import EngineConfig

__all__ = ["EngineConfig"]
