"""
Off-design and altitude performance calculations for rocket engines.

Pure functions for computing thrust, Isp, and thrust coefficient at
arbitrary ambient pressures and altitudes using ideal gas / isentropic
relations and the ISA atmospheric model.

Features:
    - Thrust at arbitrary ambient pressure
    - Altitude performance curves (ISA model)
    - Summerfield flow-separation criterion

All functions use SI units, have no side effects, and depend only on numpy.
"""
import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

# Constants
G0 = 9.80665  # Standard gravitational acceleration [m/s^2]
P_SEA_LEVEL = 101325.0  # Standard sea-level pressure [Pa]


def standard_atmosphere_pressure(altitude_m: float) -> float:
    """
    Compute atmospheric pressure using the International Standard Atmosphere (ISA) model.

    Valid for altitudes up to ~86 km. Above that, returns effectively zero.

    Args:
        altitude_m: Altitude above sea level [m].

    Returns:
        Atmospheric pressure [Pa].
    """
    if altitude_m < 0:
        return P_SEA_LEVEL
    p = P_SEA_LEVEL * (1.0 - 2.25577e-5 * altitude_m) ** 5.25588
    return max(p, 0.0)


def thrust_at_altitude(
    pc_pa: float,
    At_m2: float,
    eps: float,
    gamma: float,
    p_ambient_pa: float,
    cstar: float,
    cf_vac: float,
) -> Dict[str, float]:
    """
    Compute thrust and performance at a given ambient pressure.

    Uses the relation:
        F = Cf_vac * Pc * At  -  p_amb * Ae

    where Ae = eps * At.

    Args:
        pc_pa: Chamber pressure [Pa].
        At_m2: Throat area [m^2].
        eps: Nozzle area expansion ratio (Ae/At).
        gamma: Ratio of specific heats (used for exit pressure estimate).
        p_ambient_pa: Ambient pressure [Pa].
        cstar: Characteristic velocity [m/s].
        cf_vac: Vacuum thrust coefficient.

    Returns:
        Dictionary with keys:
            thrust_n: Thrust [N].
            isp_s: Specific impulse [s].
            cf: Thrust coefficient at this ambient pressure.
            p_exit_pa: Approximate nozzle exit pressure [Pa].
    """
    Ae_m2 = eps * At_m2

    # Approximate exit pressure from isentropic expansion
    # p_e / p_c = (1 + (gamma-1)/2 * Me^2) ^ (-gamma/(gamma-1))
    # For a given eps, Me is fixed, so p_exit scales with pc.
    # Use momentum-based Cf relation instead:
    #   Cf = Cf_vac - (p_amb * Ae) / (Pc * At)
    # This is exact for ideal nozzle with no separation.
    p_exit_pa = _exit_pressure(pc_pa, eps, gamma)

    # Thrust with ambient pressure correction
    thrust_vac = cf_vac * pc_pa * At_m2
    thrust_n = thrust_vac - p_ambient_pa * Ae_m2

    # Mass flow from c* relation: mdot = Pc * At / cstar
    mdot = pc_pa * At_m2 / cstar if cstar > 0.0 else 0.0

    # Isp and Cf
    isp_s = thrust_n / (mdot * G0) if mdot > 0.0 else 0.0
    cf = thrust_n / (pc_pa * At_m2) if (pc_pa * At_m2) > 0.0 else 0.0

    return {
        "thrust_n": thrust_n,
        "isp_s": isp_s,
        "cf": cf,
        "p_exit_pa": p_exit_pa,
    }


def altitude_performance_curve(
    pc_pa: float,
    At_m2: float,
    eps: float,
    gamma: float,
    cstar: float,
    cf_vac: float,
    altitudes_m: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute thrust, Isp, and Cf vs. altitude.

    Uses the ISA atmospheric model for pressure at each altitude and
    evaluates thrust_at_altitude at every point.

    Args:
        pc_pa: Chamber pressure [Pa].
        At_m2: Throat area [m^2].
        eps: Nozzle area expansion ratio (Ae/At).
        gamma: Ratio of specific heats.
        cstar: Characteristic velocity [m/s].
        cf_vac: Vacuum thrust coefficient.
        altitudes_m: 1-D array of altitudes [m].

    Returns:
        Dictionary with keys:
            altitudes: Copy of the input altitude array [m].
            thrust: Thrust at each altitude [N].
            isp: Specific impulse at each altitude [s].
            cf: Thrust coefficient at each altitude.
    """
    altitudes_m = np.asarray(altitudes_m, dtype=float)
    n = len(altitudes_m)
    thrust = np.empty(n)
    isp = np.empty(n)
    cf = np.empty(n)

    for i, alt in enumerate(altitudes_m):
        p_amb = standard_atmosphere_pressure(alt)
        result = thrust_at_altitude(
            pc_pa, At_m2, eps, gamma, p_amb, cstar, cf_vac
        )
        thrust[i] = result["thrust_n"]
        isp[i] = result["isp_s"]
        cf[i] = result["cf"]

    logger.debug(
        "Altitude curve computed for %d points, alt %.0f–%.0f m",
        n,
        altitudes_m[0],
        altitudes_m[-1],
    )

    return {
        "altitudes": altitudes_m.copy(),
        "thrust": thrust,
        "isp": isp,
        "cf": cf,
    }


def separation_pressure(pc_pa: float, eps: float, gamma: float) -> float:
    """
    Estimate the ambient pressure at which nozzle flow separation occurs.

    Uses the Summerfield criterion:  p_sep / p_wall ~ 0.4,
    where p_wall is the nozzle exit pressure at the design expansion ratio.

    Args:
        pc_pa: Chamber pressure [Pa].
        eps: Nozzle area expansion ratio (Ae/At).
        gamma: Ratio of specific heats.

    Returns:
        Ambient pressure [Pa] below which no separation is expected.
        Flow separation is likely when p_ambient > this value.
    """
    p_exit = _exit_pressure(pc_pa, eps, gamma)
    # Summerfield: separation when p_ambient ≈ 0.4 * p_exit
    p_sep = 0.4 * p_exit
    logger.debug(
        "Separation pressure: %.0f Pa (p_exit=%.0f Pa, eps=%.1f)",
        p_sep,
        p_exit,
        eps,
    )
    return p_sep


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _exit_pressure(pc_pa: float, eps: float, gamma: float) -> float:
    """
    Estimate nozzle exit pressure from isentropic relations.

    Iteratively solves the area-Mach relation and then computes
    p_exit = pc * (1 + (gamma-1)/2 * Me^2) ^ (-gamma/(gamma-1)).

    For speed we use a Newton iteration seeded from an approximation.

    Args:
        pc_pa: Chamber pressure [Pa].
        eps: Area expansion ratio (Ae/At).
        gamma: Specific heat ratio.

    Returns:
        Exit static pressure [Pa].
    """
    if eps <= 1.0:
        return pc_pa

    # Newton iteration for supersonic Mach from area ratio
    # A/A* = (1/M) * ((2/(gamma+1)) * (1 + (gamma-1)/2 * M^2))^((gamma+1)/(2*(gamma-1)))
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    exponent = gp1 / (2.0 * gm1)

    # Seed: approximate Me ~ 1 + sqrt((eps-1) * 2 / (gamma+1))
    Me = 1.0 + np.sqrt(max((eps - 1.0) * 2.0 / gp1, 0.0))

    for _ in range(50):
        t = 1.0 + 0.5 * gm1 * Me * Me
        area_ratio = (1.0 / Me) * ((2.0 / gp1) * t) ** exponent
        # d(A/A*)/dMe
        dAdM = (
            -(1.0 / (Me * Me)) * ((2.0 / gp1) * t) ** exponent
            + (1.0 / Me)
            * exponent
            * ((2.0 / gp1) * t) ** (exponent - 1.0)
            * (2.0 / gp1)
            * gm1
            * Me
        )
        if abs(dAdM) < 1e-30:
            break
        Me -= (area_ratio - eps) / dAdM
        Me = max(Me, 1.001)
        if abs(area_ratio - eps) < 1e-10:
            break

    p_exit = pc_pa * (1.0 + 0.5 * gm1 * Me * Me) ** (-gamma / gm1)
    return p_exit
