"""
Structural analysis functions for RESA.

Pure functions for wall thickness calculations considering pressure loads,
thermal stress, and combined stress states. No side effects.
"""
import math
from typing import Dict


def min_wall_thickness_pressure(
    p_internal_pa: float,
    radius_m: float,
    sigma_yield_pa: float,
    safety_factor: float = 2.0,
) -> float:
    """Minimum wall thickness for internal pressure (Barlow's formula).

    Args:
        p_internal_pa: Internal pressure [Pa]
        radius_m: Local chamber/nozzle radius [m]
        sigma_yield_pa: Material yield strength [Pa]
        safety_factor: Safety factor on yield

    Returns:
        Minimum wall thickness [m]
    """
    sigma_allow = sigma_yield_pa / safety_factor
    if sigma_allow <= 0:
        return float("inf")
    return p_internal_pa * radius_m / sigma_allow


def hoop_stress(
    p_internal_pa: float,
    radius_m: float,
    wall_thickness_m: float,
) -> float:
    """Hoop stress in a thin-walled pressure vessel.

    Args:
        p_internal_pa: Internal pressure [Pa]
        radius_m: Local radius [m]
        wall_thickness_m: Wall thickness [m]

    Returns:
        Hoop stress [Pa]
    """
    if wall_thickness_m <= 0:
        return float("inf")
    return p_internal_pa * radius_m / wall_thickness_m


def thermal_stress(
    delta_T: float,
    alpha_cte: float,
    E_modulus: float,
    poisson: float = 0.3,
) -> float:
    """Thermal stress from temperature gradient through wall.

    For a constrained plate/shell with temperature difference across thickness.

    Args:
        delta_T: Temperature difference across wall [K]
        alpha_cte: Coefficient of thermal expansion [1/K]
        E_modulus: Young's modulus [Pa]
        poisson: Poisson's ratio

    Returns:
        Thermal stress [Pa]
    """
    return E_modulus * alpha_cte * delta_T / (1.0 - poisson)


def min_wall_thickness_thermal(
    q_flux_w_m2: float,
    k_wall: float,
    alpha_cte: float,
    E_modulus: float,
    sigma_yield_pa: float,
    safety_factor: float = 1.5,
    poisson: float = 0.3,
) -> float:
    """Minimum wall thickness limited by thermal stress.

    The thermal stress increases with wall thickness (larger delta_T for
    the same heat flux). This finds the thickness where thermal stress
    equals the allowable stress.

    sigma_th = E * alpha * (q * t / k) / (1 - nu) = sigma_allow

    Args:
        q_flux_w_m2: Heat flux through wall [W/m^2]
        k_wall: Wall thermal conductivity [W/(m*K)]
        alpha_cte: Coefficient of thermal expansion [1/K]
        E_modulus: Young's modulus [Pa]
        sigma_yield_pa: Material yield strength [Pa]
        safety_factor: Safety factor on thermal stress
        poisson: Poisson's ratio

    Returns:
        Maximum allowable wall thickness from thermal constraints [m]
    """
    sigma_allow = sigma_yield_pa / safety_factor
    denominator = E_modulus * alpha_cte * q_flux_w_m2
    if denominator <= 0:
        return float("inf")
    return sigma_allow * k_wall * (1.0 - poisson) / denominator


def combined_wall_stress(
    p_internal_pa: float,
    radius_m: float,
    wall_thickness_m: float,
    q_flux_w_m2: float,
    k_wall: float,
    alpha_cte: float,
    E_modulus: float,
    poisson: float = 0.3,
) -> Dict[str, float]:
    """Compute combined stress state at a wall station.

    Args:
        p_internal_pa: Internal (chamber) pressure [Pa]
        radius_m: Local radius [m]
        wall_thickness_m: Wall thickness [m]
        q_flux_w_m2: Local heat flux [W/m^2]
        k_wall: Wall thermal conductivity [W/(m*K)]
        alpha_cte: Coefficient of thermal expansion [1/K]
        E_modulus: Young's modulus [Pa]
        poisson: Poisson's ratio

    Returns:
        Dict with 'hoop_stress', 'thermal_stress', 'von_mises', 'safety_factor_vm'
    """
    sigma_hoop = hoop_stress(p_internal_pa, radius_m, wall_thickness_m)

    # Temperature difference across wall from heat flux
    if k_wall > 0 and wall_thickness_m > 0:
        delta_T = q_flux_w_m2 * wall_thickness_m / k_wall
    else:
        delta_T = 0.0

    sigma_thermal = thermal_stress(delta_T, alpha_cte, E_modulus, poisson)

    # Combined principal stresses at the cold (outer) wall face — the critical location
    # where both hoop stress and thermal stress are tensile:
    #   σ_1 (circumferential) = σ_hoop + σ_thermal
    #   σ_2 (axial)           = σ_hoop/2 + σ_thermal   (thin-wall: σ_axial = σ_hoop/2)
    # Both stresses act in-plane; Von Mises for biaxial state: √(σ₁² + σ₂² − σ₁σ₂)
    # Note: thermal stress here is E·α·ΔT/(1−ν), the peak surface value for a
    # constrained plate. It acts equally in the circumferential and axial directions.
    sigma_1 = sigma_hoop + sigma_thermal
    sigma_2 = sigma_hoop / 2.0 + sigma_thermal
    sigma_vm = math.sqrt(sigma_1 ** 2 + sigma_2 ** 2 - sigma_1 * sigma_2)

    return {
        "hoop_stress": sigma_hoop,
        "thermal_stress": sigma_thermal,
        "von_mises": sigma_vm,
        "delta_T": delta_T,
    }


def fatigue_life_estimate(
    sigma_max_pa: float,
    sigma_min_pa: float,
    sigma_f_prime: float,
    epsilon_f_prime: float,
    b: float = -0.12,
    c: float = -0.6,
    E_modulus: float = 200e9,
) -> float:
    """Simplified Coffin-Manson low-cycle fatigue life estimate.

    Uses the strain-life approach:
      epsilon_a = (sigma_f'/E) * (2*Nf)^b + epsilon_f' * (2*Nf)^c

    Args:
        sigma_max_pa: Maximum cyclic stress [Pa]
        sigma_min_pa: Minimum cyclic stress [Pa]
        sigma_f_prime: Fatigue strength coefficient [Pa]
        epsilon_f_prime: Fatigue ductility coefficient
        b: Fatigue strength exponent (typically -0.12 to -0.05)
        c: Fatigue ductility exponent (typically -0.6 to -0.5)
        E_modulus: Young's modulus [Pa]

    Returns:
        Estimated cycles to failure (Nf)
    """
    sigma_a = (sigma_max_pa - sigma_min_pa) / 2.0
    if sigma_a <= 0:
        return float("inf")

    epsilon_a = sigma_a / E_modulus

    # Newton-Raphson to solve for Nf
    nf = 1000.0  # initial guess
    for _ in range(50):
        two_nf = 2.0 * nf
        if two_nf <= 0:
            return float("inf")
        f = (
            (sigma_f_prime / E_modulus) * two_nf**b
            + epsilon_f_prime * two_nf**c
            - epsilon_a
        )
        df = (
            (sigma_f_prime / E_modulus) * b * two_nf ** (b - 1) * 2.0
            + epsilon_f_prime * c * two_nf ** (c - 1) * 2.0
        )
        if abs(df) < 1e-30:
            break
        nf_new = nf - f / df
        if nf_new <= 0:
            nf_new = nf / 2.0
        if abs(nf_new - nf) / max(abs(nf), 1) < 1e-6:
            nf = nf_new
            break
        nf = nf_new

    return max(nf, 0)
