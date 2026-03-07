/**
 * TypeScript interfaces mirroring the Python Pydantic models in api/models/engine_models.py
 */

export interface EngineConfigRequest {
  // Identification
  engine_name: string;
  version: string;
  designer: string;
  description: string;

  // Propellants
  fuel: string;
  oxidizer: string;
  fuel_injection_temp_k: number;
  oxidizer_injection_temp_k: number;

  // Performance targets
  thrust_n: number;
  pc_bar: number;
  mr: number;
  eff_combustion: number;

  // Efficiencies
  eff_nozzle_divergence: number;
  freeze_at_throat: boolean;

  // Nozzle design
  nozzle_type: "bell" | "conical" | "ideal";
  throat_diameter: number;
  expansion_ratio: number;
  p_exit_bar: number;
  L_star: number;
  contraction_ratio: number;
  theta_convergent: number;
  theta_exit: number;
  bell_fraction: number;

  // Cooling system
  coolant_name: string;
  cooling_mode: "counter-flow" | "co-flow";
  coolant_mass_fraction: number;
  coolant_p_in_bar: number;
  coolant_t_in_k: number;
  channel_width_throat: number;
  channel_height: number;
  rib_width_throat: number;
  wall_thickness: number;
  wall_roughness: number;
  wall_conductivity: number;
  wall_material: string;

  // Injector
  injector_dp_bar: number;
}

export const DEFAULT_ENGINE_CONFIG: EngineConfigRequest = {
  engine_name: "Unnamed Engine",
  version: "1.0",
  designer: "",
  description: "",
  fuel: "Ethanol90",
  oxidizer: "N2O",
  fuel_injection_temp_k: 298.0,
  oxidizer_injection_temp_k: 278.0,
  thrust_n: 1000.0,
  pc_bar: 20.0,
  mr: 4.0,
  eff_combustion: 0.95,
  eff_nozzle_divergence: 0.983,
  freeze_at_throat: false,
  nozzle_type: "bell",
  throat_diameter: 0.0,
  expansion_ratio: 0.0,
  p_exit_bar: 1.013,
  L_star: 1100.0,
  contraction_ratio: 10.0,
  theta_convergent: 30.0,
  theta_exit: 15.0,
  bell_fraction: 0.8,
  coolant_name: "REFPROP::NitrousOxide",
  cooling_mode: "counter-flow",
  coolant_mass_fraction: 1.0,
  coolant_p_in_bar: 50.0,
  coolant_t_in_k: 290.0,
  channel_width_throat: 1.0e-3,
  channel_height: 0.75e-3,
  rib_width_throat: 0.6e-3,
  wall_thickness: 0.5e-3,
  wall_roughness: 20e-6,
  wall_conductivity: 15.0,
  wall_material: "inconel718",
  injector_dp_bar: 0.0,
};

export interface ValidationResponse {
  is_valid: boolean;
  errors: string[];
  warnings: string[];
}

export interface CombustionResultResponse {
  pc_bar: number;
  mr: number;
  cstar: number;
  isp_vac: number;
  isp_opt: number;
  T_combustion: number;
  gamma: number;
  mw: number;
  mach_exit: number;
}

export interface EngineDesignResponse {
  timestamp: string;
  run_type: string;
  pc_bar: number;
  mr: number;
  isp_vac: number;
  isp_sea: number;
  thrust_vac: number;
  thrust_sea: number;
  massflow_total: number;
  massflow_ox: number;
  massflow_fuel: number;
  dt_mm: number;
  de_mm: number;
  length_mm: number;
  expansion_ratio: number;
  combustion?: CombustionResultResponse;
  // Plotly JSON strings
  figure_dashboard?: string | null;
  figure_contour?: string | null;
  figure_gas_dynamics?: string | null;
  figure_3d?: string | null;
  // Cooling
  max_wall_temp?: number | null;
  max_heat_flux?: number | null;
  pressure_drop_bar?: number | null;
  outlet_temp_k?: number | null;
  // Nozzle contour
  contour_x_mm?: number[] | null;
  contour_y_mm?: number[] | null;
  warnings: string[];
}
