/**
 * Types for nozzle contour API.
 */

export interface ContourConfig {
  bell_fraction: number | null;
  theta_exit: number | null;
  resolution: number;
  wall_thickness_mm: number;
}

export const DEFAULT_CONTOUR_CONFIG: ContourConfig = {
  bell_fraction: null,
  theta_exit: null,
  resolution: 200,
  wall_thickness_mm: 1.0,
};

export interface ContourResponse {
  throat_diameter_mm: number;
  exit_diameter_mm: number;
  total_length_mm: number;
  expansion_ratio: number;
  x_mm: number[];
  y_mm: number[];
  x_chamber_mm: number[];
  y_chamber_mm: number[];
  x_convergent_mm: number[];
  y_convergent_mm: number[];
  x_divergent_mm: number[];
  y_divergent_mm: number[];
  figure_contour: string | null;
  figure_3d: string | null;
}
