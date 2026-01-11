"""
N2O saturation properties and thermodynamic calculations.

Provides functions for calculating saturation pressure, density,
latent heat, and other thermodynamic properties using CoolProp.
"""

import CoolProp.CoolProp as CP


def get_saturation_pressure(temperature: float, fluid: str = 'NitrousOxide') -> float:
    """Get saturation pressure of fluid at given temperature.

    Args:
        temperature: Temperature in K
        fluid: CoolProp fluid name (default: NitrousOxide)

    Returns:
        Saturation pressure in Pa
    """
    try:
        return CP.PropsSI('P', 'T', temperature, 'Q', 0, fluid)
    except Exception:
        return CP.PropsSI('Pcrit', fluid)


def get_liquid_density(temperature: float, fluid: str = 'NitrousOxide') -> float:
    """Get saturated liquid density at given temperature.

    Args:
        temperature: Temperature in K
        fluid: CoolProp fluid name (default: NitrousOxide)

    Returns:
        Liquid density in kg/m^3
    """
    try:
        return CP.PropsSI('D', 'T', temperature, 'Q', 0, fluid)
    except Exception:
        # Fallback to compressed liquid at saturation pressure
        P_sat = get_saturation_pressure(temperature, fluid)
        return CP.PropsSI('D', 'P', P_sat, 'T', temperature, fluid)


def get_vapor_density(temperature: float, fluid: str = 'NitrousOxide') -> float:
    """Get saturated vapor density at given temperature.

    Args:
        temperature: Temperature in K
        fluid: CoolProp fluid name (default: NitrousOxide)

    Returns:
        Vapor density in kg/m^3
    """
    try:
        return CP.PropsSI('D', 'T', temperature, 'Q', 1, fluid)
    except Exception:
        return 20.0  # Fallback ~20 kg/m^3 at room temp


def get_latent_heat(temperature: float, fluid: str = 'NitrousOxide') -> float:
    """Get latent heat of vaporization.

    Args:
        temperature: Temperature in K
        fluid: CoolProp fluid name (default: NitrousOxide)

    Returns:
        Latent heat in J/kg
    """
    h_vapor = CP.PropsSI('H', 'T', temperature, 'Q', 1, fluid)
    h_liquid = CP.PropsSI('H', 'T', temperature, 'Q', 0, fluid)
    return h_vapor - h_liquid


def get_liquid_cp(temperature: float, fluid: str = 'NitrousOxide') -> float:
    """Get liquid heat capacity at constant pressure.

    Args:
        temperature: Temperature in K
        fluid: CoolProp fluid name (default: NitrousOxide)

    Returns:
        Heat capacity in J/kg/K
    """
    try:
        return CP.PropsSI('C', 'T', temperature, 'Q', 0, fluid)
    except Exception:
        return 2000.0  # J/kg/K default


def get_gas_constant(fluid: str) -> float:
    """Get specific gas constant for a fluid.

    Args:
        fluid: CoolProp fluid name

    Returns:
        Specific gas constant in J/kg/K
    """
    return CP.PropsSI('gas_constant', fluid) / CP.PropsSI('molar_mass', fluid)


def get_critical_properties(fluid: str = 'NitrousOxide') -> dict:
    """Get critical point properties of a fluid.

    Args:
        fluid: CoolProp fluid name (default: NitrousOxide)

    Returns:
        Dictionary with critical temperature (K), pressure (Pa), and density (kg/m^3)
    """
    return {
        'T_crit': CP.PropsSI('Tcrit', fluid),
        'P_crit': CP.PropsSI('Pcrit', fluid),
        'rho_crit': CP.PropsSI('rhocrit', fluid)
    }
