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
    Calculates h_g using the Bartz Equation with corrected SI-Imperial conversions.
    """

    # 1. Convert to Imperial Inputs (Required for the empirical constant 0.026)
    Dt_in = d_throat_m * Units.METER_TO_INCH
    Pc_psi = pc_pa * Units.PA_TO_PSI
    cstar_fps = c_star_mps * Units.METER_TO_FEET

    # --- CRITICAL FIX: Viscosity Conversion ---
    # Bartz expects viscosity in lbm / (in * s)
    # 1 Pa*s = 1 kg/(m*s)
    # 1 kg = 2.20462 lbm
    # 1 m = 39.3701 in
    # Factor = 2.20462 / 39.3701 = 0.05598
    PAS_TO_LBM_IN_S = 0.05598728
    mu_imp = viscosity_gas * PAS_TO_LBM_IN_S

    # Cp: [J/kgK] -> [BTU/lbm*F]
    J_KGK_TO_BTU_LBMF = 0.000238846
    cp_imp = cp_gas * J_KGK_TO_BTU_LBMF

    # 2. Geometric Area Ratio (Dt/D)^2 = At/A
    area_ratios = (d_throat_m / diameters) ** 2

    # 3. Sigma Factor (Boundary Layer Property Correction)
    # Using fixed wall temp ratio estimation for stability
    # Ideally: Iterate with T_wall calculated by the cooling solver
    wall_temp_ratio = 0.6

    stag_term = 1 + 0.5 * (gamma - 1) * mach_numbers ** 2
    sigma_denom = (0.5 * wall_temp_ratio * stag_term + 0.5) ** 0.68 * (stag_term) ** 0.12
    sigma = 1.0 / sigma_denom

    # 4. Calculate h_g (Imperial) [BTU / (in^2 * s * F)]
    # Constant 0.026 is based on Dt in inches and viscosity in mass units
    # g0 is gravitational constant 32.174 (lbm*ft)/(lbf*s^2) for mass flux conversion
    g0_fps = 32.174

    term_geom = 0.026 / (Dt_in ** 0.2)
    term_fluid = ((mu_imp ** 0.2) * cp_imp) / (prandtl_gas ** 0.6)
    term_flow = ((Pc_psi * g0_fps) / cstar_fps) ** 0.8
    term_area = area_ratios ** 0.9

    h_g_imp = term_geom * term_fluid * term_flow * term_area * sigma

    # 5. Convert back to SI [W / (m^2 * K)]
    # 1 BTU = 1055.06 J
    # 1 in^2 = 0.00064516 m^2
    # 1 F = 5/9 K (temperature difference)

    BTU_TO_J = 1055.06
    IN2_TO_M2 = 0.00064516
    F_TO_K = 5.0 / 9.0

    # Factor ~ 2,943,735
    conversion_factor = BTU_TO_J / (IN2_TO_M2 * 1.0 * F_TO_K)

    h_g_si = h_g_imp * conversion_factor
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