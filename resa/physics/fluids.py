"""Fluid property providers for RESA."""
import CoolProp.CoolProp as CP
from typing import Dict, Any


class CoolPropFluid:
    """
    CoolProp-based fluid property provider.

    Provides thermodynamic and transport properties for coolant fluids.
    """

    def __init__(self, fluid_name: str):
        """
        Initialize fluid provider.

        Args:
            fluid_name: CoolProp fluid name (e.g., 'Ethanol', 'Water', 'N2O')
        """
        self.fluid_name = fluid_name

    def get_properties(self, T: float, P: float) -> Dict[str, float]:
        """
        Get fluid properties at given temperature and pressure.

        Args:
            T: Temperature [K]
            P: Pressure [Pa]

        Returns:
            Dictionary with density, viscosity, conductivity, cp, etc.
        """
        try:
            return {
                'density': CP.PropsSI('D', 'T', T, 'P', P, self.fluid_name),
                'viscosity': CP.PropsSI('V', 'T', T, 'P', P, self.fluid_name),
                'conductivity': CP.PropsSI('L', 'T', T, 'P', P, self.fluid_name),
                'cp': CP.PropsSI('Cpmass', 'T', T, 'P', P, self.fluid_name),
                'enthalpy': CP.PropsSI('H', 'T', T, 'P', P, self.fluid_name),
                'quality': CP.PropsSI('Q', 'T', T, 'P', P, self.fluid_name),
            }
        except Exception as e:
            raise ValueError(f"CoolProp error for {self.fluid_name} at T={T}K, P={P}Pa: {e}")

    def get_density(self, T: float, P: float) -> float:
        """Get density [kg/m^3]."""
        return CP.PropsSI('D', 'T', T, 'P', P, self.fluid_name)

    def get_viscosity(self, T: float, P: float) -> float:
        """Get dynamic viscosity [Pa*s]."""
        return CP.PropsSI('V', 'T', T, 'P', P, self.fluid_name)

    def get_conductivity(self, T: float, P: float) -> float:
        """Get thermal conductivity [W/m-K]."""
        return CP.PropsSI('L', 'T', T, 'P', P, self.fluid_name)

    def get_cp(self, T: float, P: float) -> float:
        """Get specific heat at constant pressure [J/kg-K]."""
        return CP.PropsSI('Cpmass', 'T', T, 'P', P, self.fluid_name)

    def get_prandtl(self, T: float, P: float) -> float:
        """Get Prandtl number."""
        cp = self.get_cp(T, P)
        mu = self.get_viscosity(T, P)
        k = self.get_conductivity(T, P)
        return cp * mu / k

    def get_saturation_pressure(self, T: float) -> float:
        """Get saturation pressure at temperature [Pa]."""
        return CP.PropsSI('P', 'T', T, 'Q', 0, self.fluid_name)

    def get_saturation_temperature(self, P: float) -> float:
        """Get saturation temperature at pressure [K]."""
        return CP.PropsSI('T', 'P', P, 'Q', 0, self.fluid_name)
