/**
 * Types for design optimization API.
 */

export interface DesignVariableSpec {
  name: string;
  min_val: number;
  max_val: number;
  initial: number;
}

export interface ConstraintSpec {
  output_name: string;
  type: string;
  limit: number;
}

export interface OptimizationConfig {
  variables: DesignVariableSpec[];
  objective: string;
  minimize: boolean;
  constraints: ConstraintSpec[];
  max_iterations: number;
  algorithm: string;
}

export const DEFAULT_OPTIMIZATION_CONFIG: OptimizationConfig = {
  variables: [],
  objective: "isp_vac",
  minimize: false,
  constraints: [],
  max_iterations: 50,
  algorithm: "Nelder-Mead",
};

export interface OptimizationResponse {
  optimal_variables: Record<string, number>;
  optimal_outputs: Record<string, number>;
  objective_value: number;
  n_evaluations: number;
  converged: boolean;
  message: string;
  history_iterations: number[];
  history_objective: number[];
}
