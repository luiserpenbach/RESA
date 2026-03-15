/**
 * Types for cooling channel design and analysis API.
 */

export interface CoolingChannelConfig {
  channel_type: "rectangular" | "trapezoidal";
  num_channels_override: number | null;
  height_profile: "constant" | "tapered" | "custom";
  height_throat_m: number;
  height_chamber_m: number;
  height_exit_m: number;
  taper_angle_deg: number;
  bifurcation_enabled: boolean;
  bifurcation_station_x: number;
  aspect_ratio_limit: number;
  optimization_target: "none" | "min_dp" | "min_wall_temp" | "max_margin";
}

export const DEFAULT_COOLING_CONFIG: CoolingChannelConfig = {
  channel_type: "rectangular",
  num_channels_override: null,
  height_profile: "constant",
  height_throat_m: 0.75e-3,
  height_chamber_m: 1.5e-3,
  height_exit_m: 1.0e-3,
  taper_angle_deg: 10.0,
  bifurcation_enabled: false,
  bifurcation_station_x: 0.0,
  aspect_ratio_limit: 10.0,
  optimization_target: "none",
};

export interface CoolingChannelResponse {
  num_channels: number;
  channel_type: string;
  height_profile: string;
  min_channel_width_mm: number;
  max_channel_width_mm: number;
  min_aspect_ratio: number;
  max_aspect_ratio: number;
  x_mm: number[];
  channel_width_mm: number[];
  channel_height_mm: number[];
}

export interface CoolingAnalysisResponse {
  max_wall_temp_k: number;
  max_heat_flux_mw_m2: number;
  pressure_drop_bar: number;
  outlet_temp_k: number;
  figure_thermal: string | null;
  x_mm: number[];
  t_wall_hot_k: number[];
  t_wall_cold_k: number[];
  t_coolant_k: number[];
  q_flux_mw_m2: number[];
  coolant_velocity_m_s: number[];
  coolant_pressure_bar: number[];
  warnings: string[];
}
