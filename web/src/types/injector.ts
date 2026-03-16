/**
 * Types for swirl injector design API.
 */

export interface PropellantConfigRequest {
  fuel: string;
  oxidizer: string;
  fuel_temperature: number;
  oxidizer_temperature: number;
}

export interface OperatingConditionsRequest {
  inlet_pressure: number;
  pressure_drop: number;
  mass_flow_fuel: number;
  mass_flow_oxidizer: number;
  oxidizer_velocity: number;
}

export interface GeometryConfigRequest {
  num_elements: number;
  num_fuel_ports: number;
  num_ox_orifices: number;
  post_thickness: number;
  spray_half_angle: number;
  minimum_clearance: number;
}

export interface InjectorConfig {
  injector_type: string;
  propellants: PropellantConfigRequest;
  operating: OperatingConditionsRequest;
  geometry: GeometryConfigRequest;
}

export const DEFAULT_INJECTOR_CONFIG: InjectorConfig = {
  injector_type: "LCSC",
  propellants: {
    fuel: "REFPROP::Ethanol",
    oxidizer: "REFPROP::NitrousOxide",
    fuel_temperature: 300.0,
    oxidizer_temperature: 500.0,
  },
  operating: {
    inlet_pressure: 45e5,
    pressure_drop: 20e5,
    mass_flow_fuel: 0.20,
    mass_flow_oxidizer: 0.80,
    oxidizer_velocity: 100.0,
  },
  geometry: {
    num_elements: 3,
    num_fuel_ports: 3,
    num_ox_orifices: 1,
    post_thickness: 0.5e-3,
    spray_half_angle: 60.0,
    minimum_clearance: 0.5e-3,
  },
};

export interface InjectorGeometryResponse {
  fuel_orifice_radius_mm: number;
  fuel_port_radius_mm: number;
  swirl_chamber_radius_mm: number;
  ox_outlet_radius_mm: number;
  ox_inlet_orifice_radius_mm: number;
  recess_length_mm: number;
}

export interface InjectorPerformanceResponse {
  spray_half_angle_deg: number;
  swirl_number: number;
  momentum_flux_ratio: number;
  velocity_ratio: number;
  weber_number: number;
  discharge_coefficient: number;
}

export interface InjectorMassFlowResponse {
  fuel_per_element_kg_s: number;
  oxidizer_per_element_kg_s: number;
  total_fuel_kg_s: number;
  total_oxidizer_kg_s: number;
  mixture_ratio: number;
}

export interface InjectorResponse {
  injector_type: string;
  geometry: InjectorGeometryResponse;
  performance: InjectorPerformanceResponse;
  mass_flows: InjectorMassFlowResponse;
}
