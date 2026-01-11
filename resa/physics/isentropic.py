"""
Isentropic flow relations for compressible gas dynamics.

All functions use SI units and have no external dependencies
except numpy and scipy.
"""
import numpy as np
from scipy.optimize import brentq
from typing import Dict

# Constants
R_UNIVERSAL = 8314.462618  # [J/(kmol*K)]
G0 = 9.80665  # Standard Gravity [m/s^2]


def get_expansion_ratio(pe: float, pc: float, gamma: float) -> float:
    """
    Calculate optimum expansion ratio for a given pressure ratio.

    Args:
        pe: Exit pressure [Pa]
        pc: Chamber pressure [Pa]
        gamma: Specific heat ratio

    Returns:
        Area expansion ratio (Ae/At)
    """
    if pe >= pc:
        return 1.0

    pr_term = (pe / pc) ** (1 / gamma)
    g_term = ((gamma + 1) / 2) ** (1 / (gamma - 1))
    v_term = np.sqrt(
        (gamma + 1) / (gamma - 1) * (1 - (pe / pc) ** ((gamma - 1) / gamma))
    )

    return 1 / (g_term * pr_term * v_term)


def mach_from_area_ratio(
    area_ratio: float,
    gamma: float,
    supersonic: bool = True
) -> float:
    """
    Solve the Area-Mach relation for Mach number.

    Uses Brent's method for robust root finding.

    Args:
        area_ratio: Local area / throat area (A/At)
        gamma: Specific heat ratio
        supersonic: True for divergent section, False for convergent

    Returns:
        Local Mach number
    """
    if area_ratio < 1.0:
        if np.isclose(area_ratio, 1.0, atol=1e-4):
            return 1.0
        raise ValueError(f"Area ratio cannot be less than 1.0 (got {area_ratio})")

    def residuals(M):
        if M <= 0:
            return 1e9
        term1 = 1.0 / M
        term2 = (2.0 / (gamma + 1.0)) * (1.0 + (gamma - 1.0) / 2.0 * M ** 2)
        exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        return term1 * (term2 ** exponent) - area_ratio

    if supersonic:
        try:
            return brentq(residuals, 1.0001, 20.0, xtol=1e-5)
        except ValueError:
            return 20.0
    else:
        try:
            return brentq(residuals, 1e-4, 0.9999, xtol=1e-5)
        except ValueError:
            return 0.0


def get_pressure_ratio(mach: float, gamma: float) -> float:
    """
    Calculate static/stagnation pressure ratio.

    Args:
        mach: Mach number
        gamma: Specific heat ratio

    Returns:
        P/P0 ratio
    """
    return (1 + (gamma - 1) / 2 * mach ** 2) ** (-gamma / (gamma - 1))


def get_temperature_ratio(mach: float, gamma: float) -> float:
    """
    Calculate static/stagnation temperature ratio.

    Args:
        mach: Mach number
        gamma: Specific heat ratio

    Returns:
        T/T0 ratio
    """
    return 1 / (1 + (gamma - 1) / 2 * mach ** 2)


def get_local_properties(
    M: float,
    pc: float,
    Tc: float,
    gamma: float,
    molar_mass: float
) -> Dict[str, float]:
    """
    Calculate local static properties from Mach number.

    Args:
        M: Local Mach number
        pc: Chamber pressure [Pa]
        Tc: Chamber temperature [K]
        gamma: Specific heat ratio
        molar_mass: Gas molar mass [kg/kmol]

    Returns:
        Dict with 'p', 'T', 'rho', 'velocity', 'mach'
    """
    temp_factor = 1 + (gamma - 1) / 2 * M ** 2
    T_static = Tc / temp_factor
    p_static = pc / (temp_factor ** (gamma / (gamma - 1)))

    R_specific = R_UNIVERSAL / molar_mass
    rho_static = p_static / (R_specific * T_static)
    speed_of_sound = np.sqrt(gamma * R_specific * T_static)
    velocity = M * speed_of_sound

    return {
        'p': p_static,
        'T': T_static,
        'rho': rho_static,
        'velocity': velocity,
        'mach': M
    }
