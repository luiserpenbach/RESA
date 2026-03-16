"""
Pydantic models for Monte Carlo uncertainty analysis API.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ParameterSpec(BaseModel):
    name: str
    nominal: float
    distribution: str = "normal"
    std_dev: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mode: Optional[float] = None


class MonteCarloConfigRequest(BaseModel):
    parameters: list[ParameterSpec] = []
    n_samples: int = 100
    output_names: list[str] = ["isp_vac", "thrust_vac", "combustion.cstar"]


class OutputStatistics(BaseModel):
    mean: float
    std: float
    p5: float
    p50: float
    p95: float


class MonteCarloResponse(BaseModel):
    n_samples: int
    n_failed: int
    elapsed_s: float
    statistics: dict[str, OutputStatistics]
    sensitivity: dict[str, dict[str, float]]
    output_samples: dict[str, list[float]]
