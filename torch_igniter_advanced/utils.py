"""
Utility functions and constants for torch igniter calculations.
"""

import numpy as np

# Physical constants
G0 = 9.80665  # Standard gravity (m/s^2)
R_UNIVERSAL = 8314.46  # Universal gas constant (J/kmol-K)

# Propellant properties (for reference)
ETHANOL_MOLECULAR_WEIGHT = 46.07  # kg/kmol
N2O_MOLECULAR_WEIGHT = 44.013  # kg/kmol

# Heating values
ETHANOL_LHV = 26.8e6  # Lower Heating Value (J/kg)
ETHANOL_HHV = 29.7e6  # Higher Heating Value (J/kg)

# Typical design ranges (for validation)
TYPICAL_C_STAR_RANGE = (1200, 1800)  # m/s for Ethanol/N2O
TYPICAL_FLAME_TEMP_RANGE = (2500, 3200)  # K for Ethanol/N2O
TYPICAL_L_STAR_RANGE = (0.3, 2.0)  # m for igniters


def validate_range(value: float, range_tuple: tuple, name: str,
                    warn_only: bool = True) -> bool:
    """Validate that a value falls within expected range.

    Args:
        value: Value to check
        range_tuple: (min, max) tuple defining acceptable range
        name: Parameter name for error message
        warn_only: If True, print warning instead of raising exception

    Returns:
        True if in range, False otherwise

    Raises:
        ValueError: If out of range and warn_only=False
    """
    min_val, max_val = range_tuple

    if min_val <= value <= max_val:
        return True

    msg = f"{name} = {value:.2e} is outside typical range [{min_val:.2e}, {max_val:.2e}]"

    if warn_only:
        print(f"WARNING: {msg}")
        return False
    else:
        raise ValueError(msg)


def circle_area(diameter: float) -> float:
    """Calculate area of circle from diameter.

    Args:
        diameter: Circle diameter (m)

    Returns:
        Area (m^2)
    """
    return np.pi * diameter**2 / 4


def diameter_from_area(area: float) -> float:
    """Calculate diameter of circle from area.

    Args:
        area: Circle area (m^2)

    Returns:
        Diameter (m)
    """
    return np.sqrt(4 * area / np.pi)


def conical_nozzle_length(throat_diameter: float, exit_diameter: float,
                           half_angle_deg: float) -> float:
    """Calculate length of conical nozzle section.

    Args:
        throat_diameter: Throat diameter (m)
        exit_diameter: Exit diameter (m)
        half_angle_deg: Cone half angle (degrees)

    Returns:
        Nozzle length (m)
    """
    half_angle_rad = np.deg2rad(half_angle_deg)
    radius_diff = (exit_diameter - throat_diameter) / 2
    return radius_diff / np.tan(half_angle_rad)


def mass_flow_rate_from_velocity(velocity: float, area: float,
                                  density: float) -> float:
    """Calculate mass flow rate from velocity and density.

    Args:
        velocity: Flow velocity (m/s)
        area: Flow area (m^2)
        density: Fluid density (kg/m^3)

    Returns:
        Mass flow rate (kg/s)
    """
    return velocity * area * density


def velocity_from_mass_flow(mass_flow: float, area: float,
                            density: float) -> float:
    """Calculate velocity from mass flow rate and density.

    Args:
        mass_flow: Mass flow rate (kg/s)
        area: Flow area (m^2)
        density: Fluid density (kg/m^3)

    Returns:
        Velocity (m/s)
    """
    return mass_flow / (area * density)


def format_pressure(pressure_pa: float) -> str:
    """Format pressure value with appropriate units.

    Args:
        pressure_pa: Pressure in Pa

    Returns:
        Formatted string with units
    """
    if pressure_pa > 1e6:
        return f"{pressure_pa/1e6:.2f} MPa"
    elif pressure_pa > 1e5:
        return f"{pressure_pa/1e5:.2f} bar"
    elif pressure_pa > 1e3:
        return f"{pressure_pa/1e3:.2f} kPa"
    else:
        return f"{pressure_pa:.0f} Pa"


def format_temperature(temp_k: float) -> str:
    """Format temperature value.

    Args:
        temp_k: Temperature in K

    Returns:
        Formatted string
    """
    return f"{temp_k:.1f} K ({temp_k-273.15:.1f} °C)"


def format_mass_flow(mass_flow_kgs: float) -> str:
    """Format mass flow rate with appropriate units.

    Args:
        mass_flow_kgs: Mass flow rate in kg/s

    Returns:
        Formatted string with units
    """
    if mass_flow_kgs < 0.001:
        return f"{mass_flow_kgs*1e6:.2f} mg/s"
    elif mass_flow_kgs < 1.0:
        return f"{mass_flow_kgs*1000:.2f} g/s"
    else:
        return f"{mass_flow_kgs:.3f} kg/s"