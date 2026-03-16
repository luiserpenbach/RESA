/**
 * Types for Monte Carlo uncertainty analysis API.
 */

export interface ParameterSpec {
  name: string;
  nominal: number;
  distribution: string;
  std_dev: number | null;
  min_val: number | null;
  max_val: number | null;
  mode: number | null;
}

export interface MonteCarloConfig {
  parameters: ParameterSpec[];
  n_samples: number;
  output_names: string[];
}

export const DEFAULT_MC_CONFIG: MonteCarloConfig = {
  parameters: [],
  n_samples: 100,
  output_names: ["isp_vac", "thrust_vac", "combustion.cstar"],
};

export interface OutputStatistics {
  mean: number;
  std: number;
  p5: number;
  p50: number;
  p95: number;
}

export interface MonteCarloResponse {
  n_samples: number;
  n_failed: number;
  elapsed_s: number;
  statistics: Record<string, OutputStatistics>;
  sensitivity: Record<string, Record<string, number>>;
  output_samples: Record<string, number[]>;
}
