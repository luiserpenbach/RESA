/**
 * Types for torch igniter design API.
 */

export interface IgniterConfig {
  chamber_pressure_pa: number;
  mixture_ratio: number;
  total_mass_flow_kg_s: number;
  ethanol_feed_pressure_pa: number;
  n2o_feed_pressure_pa: number;
  ethanol_feed_temperature_k: number;
  n2o_feed_temperature_k: number;
  l_star: number;
  expansion_ratio: number;
  nozzle_type: string;
  n2o_orifice_count: number;
  ethanol_orifice_count: number;
  discharge_coefficient: number;
}

export const DEFAULT_IGNITER_CONFIG: IgniterConfig = {
  chamber_pressure_pa: 20e5,
  mixture_ratio: 2.0,
  total_mass_flow_kg_s: 0.050,
  ethanol_feed_pressure_pa: 25e5,
  n2o_feed_pressure_pa: 30e5,
  ethanol_feed_temperature_k: 298.15,
  n2o_feed_temperature_k: 298.15,
  l_star: 1.0,
  expansion_ratio: 3.0,
  nozzle_type: "conical",
  n2o_orifice_count: 4,
  ethanol_orifice_count: 4,
  discharge_coefficient: 0.7,
};

export interface IgniterCombustionResponse {
  flame_temperature_k: number;
  c_star_m_s: number;
  gamma: number;
  molecular_weight: number;
  heat_power_kw: number;
}

export interface IgniterGeometryResponse {
  chamber_diameter_mm: number;
  chamber_length_mm: number;
  chamber_volume_cm3: number;
  throat_diameter_mm: number;
  exit_diameter_mm: number;
  nozzle_length_mm: number;
}

export interface IgniterInjectorResponse {
  n2o_orifice_diameter_mm: number;
  ethanol_orifice_diameter_mm: number;
  n2o_injection_velocity_m_s: number;
  ethanol_injection_velocity_m_s: number;
  n2o_pressure_drop_bar: number;
  ethanol_pressure_drop_bar: number;
}

export interface IgniterPerformanceResponse {
  isp_theoretical_s: number;
  thrust_n: number;
}

export interface IgniterMassFlowsResponse {
  total_kg_s: number;
  oxidizer_kg_s: number;
  fuel_kg_s: number;
  mixture_ratio: number;
}

export interface IgniterResponse {
  combustion: IgniterCombustionResponse;
  geometry: IgniterGeometryResponse;
  injector: IgniterInjectorResponse;
  performance: IgniterPerformanceResponse;
  mass_flows: IgniterMassFlowsResponse;
}
