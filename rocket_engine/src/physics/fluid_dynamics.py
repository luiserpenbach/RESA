import numpy as np
import CoolProp.CoolProp as CP


def friction_factor_gs(Re: float, roughness: float, Dh: float) -> float:
    """
    Goudar-Sonnad approximation for Darcy Friction Factor.
    """
    if np.any(Re < 2000):
        return 64.0 / (Re + 1e-10)

    epsilon = roughness / Dh
    a = 2.0 / np.log(10)
    b = epsilon / 3.7
    d = (np.log(10) * Re) / 5.02
    s = b * d + np.log(d)
    q = s ** (s / (s + 1))
    g = b * d + np.log(d / q)
    z = np.log(q / g)
    delta_la = (z * g) / (g + 1)
    delta_cfa = delta_la * (1 + (z / 2) / ((g + 1) ** 2 + (z / 3) * (2 * g - 1)))
    inv_sqrt_f = a * (np.log(d / q) + delta_cfa)
    return (1.0 / inv_sqrt_f) ** 2


def flow_spi(area, cd, rho, dp):
    """Single Phase Incompressible Flow"""
    if dp < 0: return 0.0
    return cd * area * np.sqrt(2 * rho * dp)


def flow_n2o_la_luna(area, p_up, p_back, t_up, fluid_name="REFPROP::NitrousOxide"):
    """
    La Luna (2022) FML flashing model for N2O Injectors.
    Vectorized for performance in transient loops.
    """
    # Quick Check: If P_up < P_sat, it's gas/2-phase entry?
    # For simplicity, assuming liquid/supercritical entry.

    # 1. Properties
    try:
        P_vap = CP.PropsSI('P', 'T', t_up, 'Q', 0, fluid_name)
        rho_l = CP.PropsSI('D', 'T', t_up, 'Q', 0, fluid_name)
        # Latent Heat h_fg
        h_l = CP.PropsSI('H', 'T', t_up, 'Q', 0, fluid_name)
        h_v = CP.PropsSI('H', 'T', t_up, 'Q', 1, fluid_name)
        h_fg = h_v - h_l
        rho_v = CP.PropsSI('D', 'T', t_up, 'Q', 1, fluid_name)
    except:
        return flow_spi(area, 0.6, 750, p_up - p_back)  # Fallback

    # 2. Integration
    # Simplified characteristic length for standard injectors
    L_eff = 0.005  # 5mm approx
    steps = 20
    x_grid = np.linspace(0, L_eff, steps)
    P_grid = np.linspace(p_up, p_back, steps)

    alpha = 0.0  # Void fraction
    k_relax = 1.0

    for i in range(1, steps):
        dp_dx = (P_grid[i] - P_grid[i - 1]) / (x_grid[i] - x_grid[i - 1])

        if P_grid[i - 1] > P_vap:
            dalpha = 0.0
        else:
            # Flashing generation term
            dalpha = k_relax * 2 * alpha * (1 - alpha) * (-dp_dx) / (rho_l * h_fg)
            # Startup nucleation assumption (alpha can't be 0 if P < Pvap)
            if alpha < 1e-6: alpha = 1e-6

        alpha += dalpha * (x_grid[i] - x_grid[i - 1])
        alpha = min(max(alpha, 0.0), 0.99)

    # 3. Exit Velocity
    rho_exit = (1 - alpha) * rho_l + alpha * rho_v
    # Bernoulli with effective density
    v_exit = np.sqrt(2 * (p_up - p_back) / (rho_l * (1 - alpha) + 1e-9))

    return rho_exit * area * v_exit