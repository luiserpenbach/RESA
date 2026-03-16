"""
Pydantic models for design optimization API.
"""
from __future__ import annotations

from pydantic import BaseModel


class DesignVariableSpec(BaseModel):
    name: str
    min_val: float
    max_val: float
    initial: float


class ConstraintSpec(BaseModel):
    output_name: str
    type: str = "max"
    limit: float


class OptimizationConfigRequest(BaseModel):
    variables: list[DesignVariableSpec] = []
    objective: str = "isp_vac"
    minimize: bool = False
    constraints: list[ConstraintSpec] = []
    max_iterations: int = 50
    algorithm: str = "Nelder-Mead"


class OptimizationResponse(BaseModel):
    optimal_variables: dict[str, float]
    optimal_outputs: dict[str, float]
    objective_value: float
    n_evaluations: int
    converged: bool
    message: str
    history_iterations: list[int]
    history_objective: list[float]
