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

  // Wall geometry overrides (null = use engine config value)
  wall_thickness_mm: number | null;
  rib_width_throat_mm: number | null;

  // Axial margins for CAD import
  start_margin_mm: number;
  end_margin_mm: number;

  // Surface roughness override (null = use engine config value)
  roughness_microns: number | null;

  // Helix/spiral angle
  helix_angle_deg: number;

  // Coolant inlet overrides (null = use engine config value)
  coolant_p_in_bar: number | null;
  coolant_t_in_k: number | null;
  coolant_mass_fraction: number | null;
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
  wall_thickness_mm: null,
  rib_width_throat_mm: null,
  start_margin_mm: 0.0,
  end_margin_mm: 0.0,
  roughness_microns: null,
  helix_angle_deg: 0.0,
  coolant_p_in_bar: null,
  coolant_t_in_k: null,
  coolant_mass_fraction: null,
};

export interface CoolingChannelResponse {
  num_channels: number;
  channel_type: string;
  height_profile: string;
  min_channel_width_mm: number;
  max_channel_width_mm: number;
  min_aspect_ratio: number;
  max_aspect_ratio: number;
  wall_thickness_mm_val: number;
  x_mm: number[];
  channel_width_mm: number[];
  channel_height_mm: number[];
  rib_width_mm: number[];
  inner_radius_mm: number[];
  figure_3d: string | null;
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
