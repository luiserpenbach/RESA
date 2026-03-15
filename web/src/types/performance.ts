/**
 * Types for performance maps API.
 */

export interface PerformanceMapConfig {
  altitude_range_min_m: number;
  altitude_range_max_m: number;
  altitude_points: number;
  throttle_range_min: number;
  throttle_range_max: number;
  throttle_points: number;
  mr_sweep_min: number;
  mr_sweep_max: number;
  mr_sweep_points: number;
}

export const DEFAULT_PERFORMANCE_CONFIG: PerformanceMapConfig = {
  altitude_range_min_m: 0,
  altitude_range_max_m: 100_000,
  altitude_points: 50,
  throttle_range_min: 0.3,
  throttle_range_max: 1.0,
  throttle_points: 15,
  mr_sweep_min: 2.0,
  mr_sweep_max: 8.0,
  mr_sweep_points: 20,
};

export interface AltitudePerformanceResponse {
  altitudes_m: number[];
  thrust_n: number[];
  isp_s: number[];
  cf: number[];
  separation_altitude_m: number | null;
  figure_altitude: string | null;
}

export interface ThrottleMapResponse {
  throttle_pcts: number[];
  pc_bar: number[];
  thrust_n: number[];
  isp_s: number[];
  figure_throttle: string | null;
}

export interface PerformanceFullResponse {
  altitude: AltitudePerformanceResponse | null;
  throttle: ThrottleMapResponse | null;
  figure_combined: string | null;
}
