"""
Fluid properties using CoolProp for Ethanol and Nitrous Oxide.

Handles density, enthalpy, and other thermodynamic properties
needed for injector sizing and feed system analysis.
"""

import warnings
from typing import Optional, Dict

try:
    import CoolProp.CoolProp as CP
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False
    warnings.warn("CoolProp not found. Install with: pip install CoolProp")


class FluidProperties:
    """Interface to CoolProp for propellant properties."""

    def __init__(self):
        """Initialize fluid properties calculator."""
        if not HAS_COOLPROP:
            raise ImportError(
                "CoolProp is required. Install with: pip install CoolProp"
            )

        # Fluid identifiers in CoolProp
        self.ETHANOL = "Ethanol"
        self.N2O = "N2O"

    def ethanol_density(self, temperature: float, pressure: float) -> float:
        """Get ethanol liquid density.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (Pa)

        Returns:
            Density (kg/m^3)
        """
        try:
            rho = CP.PropsSI('D', 'T', temperature, 'P', pressure, self.ETHANOL)
            return rho
        except Exception as e:
            # Fallback to approximate density if CoolProp fails
            warnings.warn(f"CoolProp failed for ethanol, using approximation: {e}")
            return 789.0  # kg/m^3 at STP

    def ethanol_viscosity(self, temperature: float, pressure: float) -> float:
        """Get ethanol dynamic viscosity.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (Pa)

        Returns:
            Dynamic viscosity (Pa-s)
        """
        try:
            mu = CP.PropsSI('V', 'T', temperature, 'P', pressure, self.ETHANOL)
            return mu
        except Exception as e:
            warnings.warn(f"CoolProp failed for ethanol viscosity: {e}")
            return 1.2e-3  # Pa-s at STP

    def n2o_density(self, temperature: float, pressure: float,
                    phase: str = 'liquid') -> float:
        """Get N2O density.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (Pa)
            phase: 'liquid', 'vapor', or 'auto' for saturation handling

        Returns:
            Density (kg/m^3)
        """
        try:
            if phase == 'auto':
                # Check if we're in two-phase region
                T_sat = CP.PropsSI('T', 'P', pressure, 'Q', 0, self.N2O)
                if abs(temperature - T_sat) < 1.0:
                    # At saturation, return liquid density
                    rho = CP.PropsSI('D', 'P', pressure, 'Q', 0, self.N2O)
                elif temperature < T_sat:
                    # Subcooled liquid
                    rho = CP.PropsSI('D', 'T', temperature, 'P', pressure, self.N2O)
                else:
                    # Superheated vapor
                    rho = CP.PropsSI('D', 'T', temperature, 'P', pressure, self.N2O)
            elif phase == 'liquid':
                # Force liquid phase (saturated if in two-phase region)
                try:
                    rho = CP.PropsSI('D', 'T', temperature, 'P', pressure, self.N2O)
                except Exception:
                    # If fails, get saturated liquid density
                    rho = CP.PropsSI('D', 'P', pressure, 'Q', 0, self.N2O)
            elif phase == 'vapor':
                # Force vapor phase
                try:
                    rho = CP.PropsSI('D', 'T', temperature, 'P', pressure, self.N2O)
                except Exception:
                    # If fails, get saturated vapor density
                    rho = CP.PropsSI('D', 'P', pressure, 'Q', 1, self.N2O)
            else:
                raise ValueError(f"Unknown phase: {phase}")

            return rho
        except Exception as e:
            warnings.warn(f"CoolProp failed for N2O density: {e}")
            # Fallback approximation for liquid N2O
            return 770.0  # kg/m^3 approximate

    def n2o_saturation_properties(self, temperature: float) -> Dict[str, float]:
        """Get N2O saturation properties at given temperature.

        Args:
            temperature: Temperature (K)

        Returns:
            Dictionary with:
            - P_sat: Saturation pressure (Pa)
            - rho_liquid: Liquid density (kg/m^3)
            - rho_vapor: Vapor density (kg/m^3)
            - h_liquid: Liquid enthalpy (J/kg)
            - h_vapor: Vapor enthalpy (J/kg)
            - h_fg: Latent heat (J/kg)
        """
        try:
            P_sat = CP.PropsSI('P', 'T', temperature, 'Q', 0, self.N2O)
            rho_liquid = CP.PropsSI('D', 'T', temperature, 'Q', 0, self.N2O)
            rho_vapor = CP.PropsSI('D', 'T', temperature, 'Q', 1, self.N2O)
            h_liquid = CP.PropsSI('H', 'T', temperature, 'Q', 0, self.N2O)
            h_vapor = CP.PropsSI('H', 'T', temperature, 'Q', 1, self.N2O)

            return {
                'P_sat': P_sat,
                'rho_liquid': rho_liquid,
                'rho_vapor': rho_vapor,
                'h_liquid': h_liquid,
                'h_vapor': h_vapor,
                'h_fg': h_vapor - h_liquid,
            }
        except Exception as e:
            raise ValueError(f"Failed to get N2O saturation properties: {e}")

    def n2o_critical_point(self) -> Dict[str, float]:
        """Get N2O critical point properties.

        Returns:
            Dictionary with T_crit (K) and P_crit (Pa)
        """
        try:
            T_crit = CP.PropsSI('Tcrit', self.N2O)
            P_crit = CP.PropsSI('Pcrit', self.N2O)

            return {
                'T_crit': T_crit,
                'P_crit': P_crit,
            }
        except Exception as e:
            warnings.warn(f"Failed to get N2O critical point: {e}")
            return {
                'T_crit': 309.52,  # K
                'P_crit': 7.245e6,  # Pa
            }

    def n2o_vapor_pressure(self, temperature: float) -> float:
        """Get N2O vapor pressure at given temperature.

        Args:
            temperature: Temperature (K)

        Returns:
            Vapor pressure (Pa)
        """
        try:
            P_sat = CP.PropsSI('P', 'T', temperature, 'Q', 0, self.N2O)
            return P_sat
        except Exception as e:
            warnings.warn(f"Failed to get N2O vapor pressure: {e}")
            # Approximate using Antoine equation
            # (this is a very rough approximation)
            import numpy as np
            T_C = temperature - 273.15
            log10_P_bar = 6.53 - 1237.0 / (T_C + 260.0)
            return 10**log10_P_bar * 1e5  # Convert bar to Pa

    def ethanol_enthalpy(self, temperature: float, pressure: float) -> float:
        """Get ethanol specific enthalpy.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (Pa)

        Returns:
            Specific enthalpy (J/kg)
        """
        try:
            h = CP.PropsSI('H', 'T', temperature, 'P', pressure, self.ETHANOL)
            return h
        except Exception as e:
            warnings.warn(f"CoolProp failed for ethanol enthalpy: {e}")
            # Approximate using c_p * (T - T_ref)
            c_p = 2440.0  # J/kg-K
            T_ref = 298.15  # K
            return c_p * (temperature - T_ref)

    def n2o_enthalpy(self, temperature: float, pressure: float,
                     phase: str = 'liquid') -> float:
        """Get N2O specific enthalpy.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (Pa)
            phase: 'liquid', 'vapor', or 'auto'

        Returns:
            Specific enthalpy (J/kg)
        """
        try:
            if phase == 'liquid':
                h = CP.PropsSI('H', 'T', temperature, 'P', pressure, self.N2O)
            elif phase == 'vapor':
                h = CP.PropsSI('H', 'T', temperature, 'P', pressure, self.N2O)
            else:  # auto
                # Determine phase based on saturation
                T_sat = CP.PropsSI('T', 'P', pressure, 'Q', 0, self.N2O)
                if temperature <= T_sat:
                    h = CP.PropsSI('H', 'P', pressure, 'Q', 0, self.N2O)
                else:
                    h = CP.PropsSI('H', 'T', temperature, 'P', pressure, self.N2O)

            return h
        except Exception as e:
            warnings.warn(f"CoolProp failed for N2O enthalpy: {e}")
            return 0.0  # Fallback


# Convenience functions for common operations

def get_ethanol_density(temperature: float, pressure: float) -> float:
    """Get ethanol density (convenience function).

    Args:
        temperature: Temperature (K)
        pressure: Pressure (Pa)

    Returns:
        Density (kg/m^3)
    """
    fp = FluidProperties()
    return fp.ethanol_density(temperature, pressure)


def get_n2o_density(temperature: float, pressure: float,
                    phase: str = 'auto') -> float:
    """Get N2O density (convenience function).

    Args:
        temperature: Temperature (K)
        pressure: Pressure (Pa)
        phase: 'liquid', 'vapor', or 'auto'

    Returns:
        Density (kg/m^3)
    """
    fp = FluidProperties()
    return fp.n2o_density(temperature, pressure, phase)


def get_n2o_saturation_pressure(temperature: float) -> float:
    """Get N2O saturation pressure (convenience function).

    Args:
        temperature: Temperature (K)

    Returns:
        Saturation pressure (Pa)
    """
    fp = FluidProperties()
    return fp.n2o_vapor_pressure(temperature)
