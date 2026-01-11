"""
Performance calculations for torch igniter.

Calculates key performance metrics including c*, Isp, efficiency,
and thermal power output using LHV method.
"""

import numpy as np
from typing import Dict, Optional

# Physical constants
G0 = 9.80665  # Standard gravity (m/s^2)

# Heating values
ETHANOL_LHV = 26.8e6  # Lower Heating Value (J/kg)


class PerformanceCalculator:
    """Calculate igniter performance metrics."""

    def calculate_actual_cstar(self, mass_flow: float, chamber_pressure: float,
                               throat_area: float) -> float:
        """Calculate actual c* from measured/calculated parameters.

        c* = P_c * A_t / m_dot

        Args:
            mass_flow: Total mass flow rate (kg/s)
            chamber_pressure: Chamber pressure (Pa)
            throat_area: Throat area (m^2)

        Returns:
            Actual c* (m/s)
        """
        return chamber_pressure * throat_area / mass_flow

    def calculate_cstar_efficiency(self, actual_cstar: float,
                                   theoretical_cstar: float) -> float:
        """Calculate c* efficiency (combustion efficiency indicator).

        Args:
            actual_cstar: Actual/measured c* (m/s)
            theoretical_cstar: Theoretical c* from CEA (m/s)

        Returns:
            c* efficiency (dimensionless, typically 0.85-0.98)
        """
        return actual_cstar / theoretical_cstar

    def calculate_exit_velocity(self, isp: float) -> float:
        """Calculate effective exit velocity from Isp.

        v_exit = Isp * g0

        Args:
            isp: Specific impulse (s)

        Returns:
            Exit velocity (m/s)
        """
        return isp * G0

    def calculate_thrust_simple(self, mass_flow: float, isp: float) -> float:
        """Calculate thrust from mass flow and Isp.

        F = m_dot * Isp * g0

        Args:
            mass_flow: Total mass flow rate (kg/s)
            isp: Specific impulse (s)

        Returns:
            Thrust (N)
        """
        return mass_flow * isp * G0

    def calculate_heat_power(self, fuel_mass_flow: float,
                            lhv_fuel: float = ETHANOL_LHV) -> float:
        """Calculate thermal power output using fuel LHV.

        Power = m_dot_fuel x LHV

        This is the standard method for calculating igniter heat power.

        Args:
            fuel_mass_flow: Fuel mass flow rate (kg/s)
            lhv_fuel: Lower heating value of fuel (J/kg)
                     Default: ETHANOL_LHV = 26.8 MJ/kg

        Returns:
            Thermal power output (W)
        """
        return fuel_mass_flow * lhv_fuel

    def calculate_characteristic_time(self, chamber_volume: float,
                                      throat_area: float,
                                      c_star: float) -> float:
        """Calculate characteristic residence time.

        tau = V_chamber / (A_throat * c*)

        Args:
            chamber_volume: Chamber volume (m^3)
            throat_area: Throat area (m^2)
            c_star: Characteristic velocity (m/s)

        Returns:
            Characteristic time (s)
        """
        return chamber_volume / (throat_area * c_star)

    def calculate_heat_flux_estimate(self, heat_power: float,
                                     chamber_surface_area: float) -> float:
        """Estimate average heat flux to chamber walls.

        This is a rough estimate assuming uniform heat distribution.
        Actual heat flux will vary significantly with location.

        Args:
            heat_power: Total thermal power (W)
            chamber_surface_area: Chamber internal surface area (m^2)

        Returns:
            Average heat flux (W/m^2)
        """
        return heat_power / chamber_surface_area

    def calculate_chamber_surface_area(self, chamber_diameter: float,
                                       chamber_length: float) -> float:
        """Calculate internal surface area of cylindrical chamber.

        Args:
            chamber_diameter: Chamber diameter (m)
            chamber_length: Chamber length (m)

        Returns:
            Surface area (m^2) including ends
        """
        # Cylindrical surface
        cylinder_area = np.pi * chamber_diameter * chamber_length

        # Two end caps
        end_area = 2 * np.pi * (chamber_diameter / 2)**2

        return cylinder_area + end_area


def calculate_all_performance_metrics(
    mass_flow: float,
    chamber_pressure: float,
    throat_area: float,
    chamber_volume: float,
    chamber_diameter: float,
    chamber_length: float,
    c_star_theoretical: float,
    isp_theoretical: float,
    mixture_ratio: float,
    ambient_pressure: float = 101325.0,
    lhv_fuel: float = ETHANOL_LHV
) -> Dict[str, float]:
    """Calculate all performance metrics in one function.

    Args:
        mass_flow: Total mass flow rate (kg/s)
        chamber_pressure: Chamber pressure (Pa)
        throat_area: Throat area (m^2)
        chamber_volume: Chamber volume (m^3)
        chamber_diameter: Chamber diameter (m)
        chamber_length: Chamber length (m)
        c_star_theoretical: Theoretical c* from CEA (m/s)
        isp_theoretical: Theoretical Isp from CEA (s)
        mixture_ratio: O/F mass ratio
        ambient_pressure: Ambient pressure (Pa)
        lhv_fuel: Fuel lower heating value (J/kg)

    Returns:
        Dictionary with all performance metrics
    """
    perf = PerformanceCalculator()

    # Calculate actual c*
    c_star_actual = perf.calculate_actual_cstar(
        mass_flow, chamber_pressure, throat_area
    )

    # c* efficiency
    c_star_eff = perf.calculate_cstar_efficiency(
        c_star_actual, c_star_theoretical
    )

    # Thrust (simplified)
    thrust = perf.calculate_thrust_simple(mass_flow, isp_theoretical)

    # Heat power using LHV method
    fuel_mass_flow = mass_flow / (1 + mixture_ratio)
    heat_power = perf.calculate_heat_power(fuel_mass_flow, lhv_fuel)

    # Characteristic time
    tau = perf.calculate_characteristic_time(
        chamber_volume, throat_area, c_star_theoretical
    )

    # Heat flux estimate
    chamber_area = perf.calculate_chamber_surface_area(
        chamber_diameter, chamber_length
    )
    heat_flux = perf.calculate_heat_flux_estimate(heat_power, chamber_area)

    # L*
    l_star = chamber_volume / throat_area

    return {
        'c_star_actual': c_star_actual,
        'c_star_efficiency': c_star_eff,
        'thrust': thrust,
        'heat_power': heat_power,
        'characteristic_time': tau,
        'heat_flux_avg': heat_flux,
        'l_star': l_star,
    }


def estimate_ignition_energy(heat_power: float, duration: float) -> float:
    """Estimate total ignition energy delivered.

    Args:
        heat_power: Thermal power output (W)
        duration: Igniter firing duration (s)

    Returns:
        Total energy (J)
    """
    return heat_power * duration


def calculate_propellant_mass(mass_flow: float, duration: float) -> float:
    """Calculate total propellant mass consumed.

    Args:
        mass_flow: Total mass flow rate (kg/s)
        duration: Duration (s)

    Returns:
        Total propellant mass (kg)
    """
    return mass_flow * duration
