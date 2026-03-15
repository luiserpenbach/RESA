from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class FeedSystemConfigRequest(BaseModel):
    feed_type: str = "pressure-fed"
    cycle_type: str = "none"
    ox_line_length_m: float = 2.0
    ox_line_diameter_m: float = 0.012
    ox_line_roughness_m: float = 15e-6
    ox_k_fittings: float = 5.0
    fuel_line_length_m: float = 1.5
    fuel_line_diameter_m: float = 0.010
    fuel_line_roughness_m: float = 15e-6
    fuel_k_fittings: float = 4.0
    tank_pressure_bar: float = 0.0
    ullage_pressure_bar: float = 5.0
    pump_efficiency: float = 0.65
    turbine_efficiency: float = 0.60
    gg_temperature_k: float = 900.0
    gg_mr: float = 0.3
    suction_head_m: float = 1.0


class FeedSystemResponse(BaseModel):
    feed_type: str
    cycle_type: str
    # Pressure budget
    required_feed_pressure_bar: float
    injector_dp_bar: float
    cooling_dp_bar: float
    line_losses_ox_bar: float
    line_losses_fuel_bar: float
    # Pump (if pump-fed)
    pump_power_ox_w: float = 0.0
    pump_power_fuel_w: float = 0.0
    pump_head_ox_m: float = 0.0
    pump_head_fuel_m: float = 0.0
    npsh_available_m: float = 0.0
    # Cycle (if turbopump)
    turbine_power_w: float = 0.0
    power_balance_margin_pct: float = 0.0
    # Figure
    figure_pressure_budget: Optional[str] = None
    warnings: list[str] = []
