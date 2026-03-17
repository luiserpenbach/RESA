"""
Heat transfer calculations for rocket engine thermal analysis.

Includes Bartz equation for convective heat transfer coefficients
and adiabatic wall temperature calculations.
"""
import numpy as np


def calculate_bartz_coefficient(
    pc: float,
    cstar: float,
    Dt: float,
    At: float,
    gamma: float,
    mw: float,
    T_c: float,
    local_area_ratio: float,
    mach: float,
    viscosity: float = None,
    cp: float = None,
    prandtl: float = 0.7,
    wall_temp_ratio: float = 0.3,
) -> float:
    """
    Calculate convective heat transfer coefficient using Bartz equation.

    Implements the SI form of the Bartz (1957) equation:
        h_g = (0.026/Dt^0.2) * (mu^0.2 * cp / Pr^0.6) * (pc*g0/c*)^0.8 * (At/A)^0.9 * sigma

    The g0 factor (9.80665 m/s²) is required for dimensional consistency when
    pc [Pa] and c* [m/s] are in SI units.

    Args:
        pc: Chamber pressure [Pa]
        cstar: Characteristic velocity [m/s]
        Dt: Throat diameter [m]
        At: Throat area [m²]
        gamma: Specific heat ratio
        mw: Molecular weight [kg/kmol]
        T_c: Combustion temperature [K]
        local_area_ratio: Local A/At
        mach: Local Mach number
        viscosity: Gas viscosity [Pa·s] (optional, estimated via Sutherland if None)
        cp: Specific heat [J/(kg·K)] (optional, estimated if None)
        prandtl: Prandtl number (default 0.7)
        wall_temp_ratio: T_wall / T_c estimate for sigma correction (default 0.3).
            For typical regen-cooled chambers (T_c~3000K, T_wall~700-900K) the
            ratio is 0.20–0.30. Pass the converged value from cooling analysis
            for best accuracy.

    Returns:
        Heat transfer coefficient [W/(m²·K)]
    """
    # Estimate properties if not provided
    R_specific = 8314.46 / mw  # J/(kg·K)

    if cp is None:
        cp = gamma * R_specific / (gamma - 1)

    if viscosity is None:
        # Sutherland's law approximation for combustion gases
        mu_ref = 1.8e-5  # Reference viscosity at 300K
        T_ref = 300.0
        S = 110.0  # Sutherland constant
        viscosity = mu_ref * (T_c / T_ref) ** 1.5 * (T_ref + S) / (T_c + S)

    # Standard gravitational acceleration — required for SI dimensional consistency
    g0 = 9.80665  # m/s²

    # Sigma factor (boundary layer correction for property variation near wall)
    stag_term = 1 + 0.5 * (gamma - 1) * mach ** 2
    sigma = 1.0 / ((0.5 * wall_temp_ratio * stag_term + 0.5) ** 0.68 * stag_term ** 0.12)

    # Bartz equation in SI units
    # term1: geometry term  [m^-0.2]
    # term2: fluid property term  [Pa^0.2·m^-0.2·K^-1] (effectively)
    # term3: momentum flux term — MUST include g0 for SI correctness  [kg^0.8·m^-1.6·s^-0.6]
    # term4: area contraction  [dimensionless]
    term1 = 0.026 / (Dt ** 0.2)
    term2 = (viscosity ** 0.2 * cp) / (prandtl ** 0.6)
    term3 = (pc * g0 / cstar) ** 0.8
    term4 = (1.0 / local_area_ratio) ** 0.9

    h_g = term1 * term2 * term3 * term4 * sigma

    # Sanity-check bounds (typical range: throat ~20,000–80,000 W/m²K for
    # small engines; far from throat can drop to ~1,000 W/m²K)
    h_g = max(500.0, min(h_g, 200_000.0))

    return h_g


def calculate_adiabatic_wall_temp(
    T_combustion: float,
    mach: float,
    gamma: float,
    prandtl: float = 0.7
) -> float:
    """
    Calculate adiabatic wall (recovery) temperature.

    Args:
        T_combustion: Combustion/stagnation temperature [K]
        mach: Local Mach number
        gamma: Specific heat ratio
        prandtl: Prandtl number (default 0.7)

    Returns:
        Adiabatic wall temperature [K]
    """
    # Recovery factor for turbulent flow
    r = prandtl ** (1/3)

    # Static temperature
    term_iso = 1 + 0.5 * (gamma - 1) * mach ** 2
    T_static = T_combustion / term_iso

    # Recovery temperature
    T_aw = T_static * (1 + r * 0.5 * (gamma - 1) * mach ** 2)

    return T_aw


def calculate_wall_temperatures(
    T_gas: np.ndarray,
    h_gas: np.ndarray,
    q_flux: np.ndarray,
    wall_thickness: float,
    wall_conductivity: float
) -> tuple:
    """
    Calculate hot and cold wall temperatures.

    Args:
        T_gas: Gas recovery temperature array [K]
        h_gas: Gas-side heat transfer coefficient [W/(m²·K)]
        q_flux: Heat flux array [W/m²]
        wall_thickness: Wall thickness [m]
        wall_conductivity: Thermal conductivity [W/(m·K)]

    Returns:
        Tuple of (T_wall_hot, T_wall_cold) arrays [K]
    """
    # Hot wall temperature from heat flux
    T_wall_hot = T_gas - q_flux / h_gas

    # Cold wall temperature (linear conduction)
    T_wall_cold = T_wall_hot - q_flux * wall_thickness / wall_conductivity

    return T_wall_hot, T_wall_cold
