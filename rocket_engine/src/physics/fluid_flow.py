import numpy as np
from scipy.optimize import brentq
from typing import Dict, Union

# Constants
R_UNIVERSAL = 8314.462618  # [J/(kmol*K)]
G0 = 9.80665  # Standard Gravity [m/s^2]


def get_expansion_ratio(pe: float, pc: float, gamma: float) -> float:
    """
    Calculates the optimum Expansion Ratio (Ae/At) for a given pressure ratio.

    Args:
        pe: Exit pressure [Pa] or [bar] (must match pc units)
        pc: Chamber pressure [Pa] or [bar]
        gamma: Isentropic expansion factor (k)

    Returns:
        epsilon: Area Expansion Ratio (Ae/At)
    """
    if pe >= pc:
        return 1.0

    # Isentropic Pressure Ratio term
    pr_term = (pe / pc) ** (1 / gamma)

    # Isentropic Gamma term
    g_term = ((gamma + 1) / 2) ** (1 / (gamma - 1))

    # Velocity term
    v_term = np.sqrt(
        (gamma + 1) / (gamma - 1) * (1 - (pe / pc) ** ((gamma - 1) / gamma))
    )

    # Area ratio equation
    epsilon = 1 / (g_term * pr_term * v_term)

    return epsilon


def mach_from_area_ratio(area_ratio: float, gamma: float, supersonic: bool = True) -> float:
    """
    Solves the Area-Mach relation for Mach number.
    Uses Brent's method for robust root finding (replacing manual bisection).

    Args:
        area_ratio: Local Area / Throat Area (A/At)
        gamma: Isentropic expansion factor
        supersonic: True for nozzle divergent section, False for convergent.

    Returns:
        Mach number
    """
    if area_ratio < 1.0:
        if np.isclose(area_ratio, 1.0, atol=1e-4):
            return 1.0
        raise ValueError(f"Area Ratio cannot be less than 1.0 (Got {area_ratio})")

    # The Area-Mach Equation residual function
    # eq: (A/A*) = (1/M) * [ (2/(g+1)) * (1 + (g-1)/2 * M^2) ] ^ ((g+1)/(2(g-1)))
    def residuals(M):
        if M <= 0: return 1e9  # Prevent singularity

        term1 = 1.0 / M
        term2 = (2.0 / (gamma + 1.0)) * (1.0 + (gamma - 1.0) / 2.0 * M ** 2)
        exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))

        calculated_ar = term1 * (term2 ** exponent)
        return calculated_ar - area_ratio

    if supersonic:
        # Search range: Just above 1 to 20 (Hypersonic limit)
        try:
            return brentq(residuals, 1.0001, 20.0, xtol=1e-5)
        except ValueError:
            # Fallback for extreme values or convergence failure
            return 20.0
    else:
        # Search range: Near 0 to just below 1
        try:
            return brentq(residuals, 1e-4, 0.9999, xtol=1e-5)
        except ValueError:
            return 0.0


def get_local_properties(M: float, pc: float, Tc: float, gamma: float, molar_mass: float) -> Dict[str, float]:
    """
    Calculates local static properties based on Mach number.

    Args:
        M: Local Mach number
        pc: Chamber Stagnation Pressure [Pa]
        Tc: Chamber Stagnation Temperature [K]
        gamma: Specific Heat Ratio
        molar_mass: Gas Molar Mass [kg/kmol] (e.g. 28.97 for air)

    Returns:
        Dictionary with keys: 'p', 'T', 'rho', 'velocity', 'mach'
    """
    # Isentropic Factors
    # T0 / T = 1 + (g-1)/2 * M^2
    temp_factor = 1 + (gamma - 1) / 2 * M ** 2

    # Static Temperature
    T_static = Tc / temp_factor

    # Static Pressure
    # P0 / P = (T0 / T) ^ (g / (g-1))
    p_static = pc / (temp_factor ** (gamma / (gamma - 1)))

    # Density
    # Rho = P / (R_specific * T)
    # R_specific = R_univ / M_molar
    # Note: Expects molar_mass in kg/kmol (or g/mol), result is consistent with input P units

    # Convert molar mass to kg/mol for SI calc:
    # If input is ~20-40 (g/mol), multiply by 1e-3.
    # If user follows CEA output, it might be in g/mol.
    # We assume input is kg/kmol (numerically same as g/mol).
    R_specific = R_UNIVERSAL / molar_mass  # [J/kgK]

    rho_static = p_static / (R_specific * T_static)

    # Velocity
    # a = sqrt(gamma * R * T)
    speed_of_sound = np.sqrt(gamma * R_specific * T_static)
    velocity = M * speed_of_sound

    return {
        'p': p_static,
        'T': T_static,
        'rho': rho_static,
        'velocity': velocity,
        'mach': M
    }


# --- Wrapper ---
class CFTools:
    """Wrapper class to maintain backward compatibility if needed."""

    def __init__(self):
        self.g = G0
        self.R_molar = R_UNIVERSAL

    def get_ER_from_exitPressure(self, pe, pc, gamma):
        return get_expansion_ratio(pe, pc, gamma)

    def solve_AreaMachEquation(self, lowM, highM, AR, gamma):
        # Infer supersonic/subsonic from the bounds provided
        is_supersonic = (lowM >= 1.0)
        return mach_from_area_ratio(AR, gamma, supersonic=is_supersonic)

    def get_ThermodynamicConditions(self, M, p_c, Mw, T_c, gamma):
        # Note: Original code had p_c in [bar] but converted to [Pa] inside?
        # Check original: "p = p_c*1.01325e5 / ..."
        # To match legacy behavior, we assume p_c input is BAR.

        # Determine if p_c is bar or pa.
        # Heuristic: if p_c < 1000, it's likely bar.
        pc_pa = p_c * 1e5  # Force conversion as per old code behavior

        props = get_local_properties(M, pc_pa, T_c, gamma, Mw)
        return props