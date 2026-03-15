/**
 * Types for feed system analysis API.
 */

export interface FeedSystemConfig {
  feed_type: "pressure-fed" | "pump-fed";
  cycle_type: "none" | "gas-generator" | "expander" | "staged-combustion";
  ox_line_length_m: number;
  ox_line_diameter_m: number;
  ox_line_roughness_m: number;
  ox_k_fittings: number;
  fuel_line_length_m: number;
  fuel_line_diameter_m: number;
  fuel_line_roughness_m: number;
  fuel_k_fittings: number;
  tank_pressure_bar: number;
  ullage_pressure_bar: number;
  pump_efficiency: number;
  turbine_efficiency: number;
  gg_temperature_k: number;
  gg_mr: number;
  suction_head_m: number;
}

export const DEFAULT_FEED_SYSTEM_CONFIG: FeedSystemConfig = {
  feed_type: "pressure-fed",
  cycle_type: "none",
  ox_line_length_m: 2.0,
  ox_line_diameter_m: 0.012,
  ox_line_roughness_m: 15e-6,
  ox_k_fittings: 5.0,
  fuel_line_length_m: 1.5,
  fuel_line_diameter_m: 0.010,
  fuel_line_roughness_m: 15e-6,
  fuel_k_fittings: 4.0,
  tank_pressure_bar: 0.0,
  ullage_pressure_bar: 5.0,
  pump_efficiency: 0.65,
  turbine_efficiency: 0.60,
  gg_temperature_k: 900.0,
  gg_mr: 0.3,
  suction_head_m: 1.0,
};

export interface FeedSystemResponse {
  feed_type: string;
  cycle_type: string;
  required_feed_pressure_bar: number;
  injector_dp_bar: number;
  cooling_dp_bar: number;
  line_losses_ox_bar: number;
  line_losses_fuel_bar: number;
  pump_power_ox_w: number;
  pump_power_fuel_w: number;
  pump_head_ox_m: number;
  pump_head_fuel_m: number;
  npsh_available_m: number;
  turbine_power_w: number;
  power_balance_margin_pct: number;
  figure_pressure_budget: string | null;
  warnings: string[];
}
