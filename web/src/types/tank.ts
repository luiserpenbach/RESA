/**
 * Types for tank simulation API.
 */

export interface TankConfigRequest {
  volume: number;
  initial_liquid_mass: number;
  initial_ullage_pressure: number;
  initial_temperature: number;
  ambient_temperature: number;
  heat_transfer_coefficient: number;
}

export interface PressurantConfigRequest {
  fluid_name: string;
  supply_pressure: number;
  supply_temperature: number;
  regulator_flow_coefficient: number;
}

export interface PropellantConfigRequest {
  fluid_name: string;
  mass_flow_rate: number;
  is_self_pressurizing: boolean;
}

export interface TankSimConfig {
  tank_type: string;
  tank: TankConfigRequest;
  pressurant: PressurantConfigRequest;
  propellant: PropellantConfigRequest;
  duration_s: number;
}

export const DEFAULT_TANK_CONFIG: TankSimConfig = {
  tank_type: "n2o",
  tank: {
    volume: 0.020,
    initial_liquid_mass: 15.0,
    initial_ullage_pressure: 60e5,
    initial_temperature: 293.15,
    ambient_temperature: 293.15,
    heat_transfer_coefficient: 5.0,
  },
  pressurant: {
    fluid_name: "Nitrogen",
    supply_pressure: 65e5,
    supply_temperature: 293.15,
    regulator_flow_coefficient: 1e-6,
  },
  propellant: {
    fluid_name: "NitrousOxide",
    mass_flow_rate: 0.8,
    is_self_pressurizing: true,
  },
  duration_s: 30.0,
};

export interface TankSimResponse {
  time_s: number[];
  pressure_bar: number[];
  liquid_mass_kg: number[];
  liquid_temperature_k: number[];
  ullage_temperature_k: number[];
  burn_duration_s: number;
  final_liquid_mass_kg: number;
  final_pressure_bar: number;
}
