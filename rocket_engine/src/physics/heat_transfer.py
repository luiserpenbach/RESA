import numpy as np
from utils.units import Units


def calculate_bartz_coefficient(
        diameters: np.ndarray,
        mach_numbers: np.ndarray,
        pc_pa: float,
        c_star_mps: float,
        d_throat_m: float,
        T_combustion: float,
        viscosity_gas: float,
        cp_gas: float,
        prandtl_gas: float,
        gamma: float
) -> np.ndarray:
    """
    Calculates the Gas-Side Heat Transfer Coefficient (h_g) using the Bartz Equation.

    Args:
        diameters: Array of local diameters [m]
        mach_numbers: Array of local Mach numbers
        pc_pa: Chamber Pressure [Pa]
        c_star_mps: Characteristic velocity [m/s]
        d_throat_m: Throat diameter [m]
        viscosity_gas: Gas viscosity in chamber [Pa*s]
        cp_gas: Gas specific heat [J/kgK]
        prandtl_gas: Gas Prandtl number
        gamma: Specific Heat Ratio

    Returns:
        h_g: Array of heat transfer coefficients [W/(m^2 K)]
    """

    # 1. Convert to Imperial (Bartz constants are imperial)
    Dt_in = d_throat_m * Units.METER_TO_INCH
    Pc_psi = pc_pa * Units.PA_TO_PSI
    cstar_fps = c_star_mps * Units.METER_TO_FEET

    # Viscosity: [Pa*s] -> [lb/(in*s)]
    # 1 Pa*s = 0.000145038 lb/(in*s)
    PAS_TO_LB_IN_S = 0.0001450377
    mu_imp = viscosity_gas * PAS_TO_LB_IN_S

    # Cp: [J/kgK] -> [BTU/lb*F]
    # 1 J/kgK = 0.000238846 BTU/lb*F
    J_KGK_TO_BTU_LBF = 0.000238846
    cp_imp = cp_gas * J_KGK_TO_BTU_LBF

    # 2. Area Ratio term (D_throat / D_local)^2 -> (A_throat / A_local)
    # The term in Bartz is (A_t / A)^0.9.
    # A_t / A = (Dt / D)^2
    area_ratios = (d_throat_m / diameters) ** 2

    # 3. Sigma Factor (Boundary Layer Correction)
    # Simplified sigma calculation
    # sigma = [ (0.5 * Tw/T0 * (1 + (g-1)/2 M^2) + 0.5)^0.68 * (1 + (g-1)/2 M^2)^0.12 ]^-1
    # Assuming Tw/T0 approx 0.8 for initial sizing (wall is hot but cooler than gas)
    wall_temp_ratio = 0.6  # Guessing Tw ~ 1500-2000K, Tc ~ 3000K

    stag_term = 1 + 0.5 * (gamma - 1) * mach_numbers ** 2
    sigma_denom = (0.5 * wall_temp_ratio * stag_term + 0.5) ** 0.68 * (stag_term) ** 0.12
    sigma = 1.0 / sigma_denom

    # 4. Calculate h_g (Imperial)
    # hg = [0.026 / Dt^0.2] * [ (mu^0.2 * Cp) / Pr^0.6 ] * [ (Pc * g0) / cstar ]^0.8 * (At/A)^0.9 * sigma

    g0_fps = 32.174

    term_geom = 0.026 / (Dt_in ** 0.2)
    term_fluid = ((mu_imp ** 0.2) * cp_imp) / (prandtl_gas ** 0.6)
    term_flow = ((Pc_psi * g0_fps) / cstar_fps) ** 0.8
    term_area = area_ratios ** 0.9

    h_g_imp = term_geom * term_fluid * term_flow * term_area * sigma

    # 5. Convert back to SI [W/m^2 K]
    # 1 BTU/(in^2 s F) = 294370 W/(m^2 K) ? No, standard Bartz output is BTU/(in^2 * s * F)
    # Wait, usually Bartz gives BTU/(in^2 s F).
    # Conversion: 1 BTU = 1055 J. 1 in^2 = 0.000645 m^2. 1 F = 5/9 K.
    # Factor = 1055 / (0.000645 * 1 * 5/9) = ~2,940,000 ??

    # Let's double check unit of output.
    # If standard bartz: h [BTU / (in^2 * sec * F)]
    BTU_TO_J = 1055.06
    IN2_TO_M2 = 0.00064516
    F_TO_K = 5.0 / 9.0

    # h_si = h_imp * [J] / ([m2] * [s] * [K])
    factor = BTU_TO_J / (IN2_TO_M2 * 1.0 * F_TO_K)  # = 2,943,735

    h_g_si = h_g_imp * factor

    return h_g_si


def calculate_adiabatic_wall_temp(
        T_combustion: float,
        gamma: float,
        mach_numbers: np.ndarray,
        prandtl: float = 0.7
) -> np.ndarray:
    """Calculates T_aw (Recovery Temperature) along the nozzle."""
    r = prandtl ** (1 / 3)  # Recovery factor for turbulent flow

    # T_static = T0 / (1 + (g-1)/2 M^2)
    # T_aw = T_static * (1 + r*(g-1)/2 M^2)

    term_iso = 1 + 0.5 * (gamma - 1) * mach_numbers ** 2
    t_static = T_combustion / term_iso

    t_aw = t_static * (1 + r * 0.5 * (gamma - 1) * mach_numbers ** 2)
    return t_aw