from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PerformanceMapConfigRequest(BaseModel):
    altitude_range_min_m: float = 0
    altitude_range_max_m: float = 100_000
    altitude_points: int = 50
    throttle_range_min: float = 0.3
    throttle_range_max: float = 1.0
    throttle_points: int = 15
    mr_sweep_min: float = 2.0
    mr_sweep_max: float = 8.0
    mr_sweep_points: int = 20


class AltitudePerformanceResponse(BaseModel):
    altitudes_m: list[float]
    thrust_n: list[float]
    isp_s: list[float]
    cf: list[float]
    separation_altitude_m: Optional[float] = None
    figure_altitude: Optional[str] = None


class ThrottleMapResponse(BaseModel):
    throttle_pcts: list[float]
    pc_bar: list[float]
    thrust_n: list[float]
    isp_s: list[float]
    figure_throttle: Optional[str] = None


class PerformanceFullResponse(BaseModel):
    altitude: Optional[AltitudePerformanceResponse] = None
    throttle: Optional[ThrottleMapResponse] = None
    figure_combined: Optional[str] = None
