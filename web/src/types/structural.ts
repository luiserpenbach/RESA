/**
 * Types for structural / wall thickness analysis API.
 */

export interface WallThicknessConfig {
  material_name: string;
  safety_factor_pressure: number;
  safety_factor_thermal: number;
  design_life_cycles: number;
  include_fatigue: boolean;
}

export const DEFAULT_WALL_THICKNESS_CONFIG: WallThicknessConfig = {
  material_name: "inconel718",
  safety_factor_pressure: 2.0,
  safety_factor_thermal: 1.5,
  design_life_cycles: 100,
  include_fatigue: false,
};

export interface MaterialPropertiesResponse {
  name: string;
  density_kg_m3: number;
  yield_strength_mpa: number;
  thermal_conductivity_w_mk: number;
  max_service_temp_k: number;
}

export interface WallThicknessResponse {
  min_safety_factor: number;
  critical_station_x_mm: number;
  max_hoop_stress_mpa: number;
  max_thermal_stress_mpa: number;
  max_von_mises_mpa: number;
  material: MaterialPropertiesResponse;
  figure_stress: string | null;
  figure_safety_factor: string | null;
  x_mm: number[];
  min_thickness_pressure_mm: number[];
  min_thickness_thermal_mm: number[];
  actual_thickness_mm: number[];
  safety_factor: number[];
  hoop_stress_mpa: number[];
  thermal_stress_mpa: number[];
  von_mises_mpa: number[];
  warnings: string[];
}

export interface MaterialListResponse {
  materials: Record<string, string>;
}
