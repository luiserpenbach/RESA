import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

# =============================================================================
# Transient startup model for pressure-fed N₂O hybrid/bipropellant engine
# Using La Luna (2022) FML flashing model for oxidizer injector
# =============================================================================

# Engine parameters (example: 5–10 kN class N₂O/HTPB or N₂O/ethanol hybrid)
c_star = 1350.0               # [m/s] typical for N₂O hybrids at O/F ~6–8
D_t = 13e-3
A_t = np.pi/4 * D_t**2                  # throat area [m²] → ~30–35 bar nominal
D_c = 100e-3
L_c = 150e-3
V_c = np.pi/4 * D_c**2 * L_c                   # chamber volume [m³]
gamma = 1.25
R_gas = 8314 / 44.013         # N₂O combustion products ≈ CO₂/H₂O/N₂ mix
T_c = 2950.0                  # [K]

# Propellant properties (fuel = ethanol for biprop, or negligible for pure hybrid)
rho_f = 789.0                 # ethanol liquid density [kg/m³] (or HTPB grain regression ignored here)

# Injector geometry (fixed)
D_ox = 0.5e-3                 # oxidizer orifice diameter [m]
N_ox = 15                     # number of oxidizer orifices
L_over_D_ox = 1.0
CdA_ox = N_ox * np.pi * (D_ox/2)**2   # geometric area

D_f = 0.35e-3                  # fuel orifices (if biprop)
N_f = 9
CdA_f = 0.65 * N_f * np.pi * (D_f/2)**2   # fuel uses normal Cd = 0.65

# Feed system
P_tank = 75e5                 # 55 bar constant tank pressure (helium press)
L_inert_ox = 8.0e6            # oxidizer line inertance [Pa·s²/kg]
L_inert_f  = 6.0e6            # fuel line inertance

# Combustion time lag (droplet evaporation + mixing)
tau_comb = 0.008              # 8 ms

# N₂O upstream temperature
T_n2o = 400.0                 # 22 °C (warm, self-pressurized)

fluid_n2o = "REFPROP::NitrousOxide"

# =============================================================================
# La Luna FML oxidizer flow solver (fast version for ODE)
# =============================================================================
def n2o_massflow_la_luna(P_up, P_back, T_up=T_n2o):
    """Single-orifice N₂O mass flow using La Luna FML (vectorized for speed)"""
    props = {
        'P_vap': PropsSI('P', 'T', T_up, 'Q', 0, fluid_n2o),
        'rho_l': PropsSI('D', 'T', T_up, 'Q', 0, fluid_n2o),
        'h_fg' : PropsSI('H', 'T', T_up, 'Q', 1, fluid_n2o) - PropsSI('H', 'T', T_up, 'Q', 0, fluid_n2o),
        'rho_v': PropsSI('D', 'T', T_up, 'Q', 1, fluid_n2o)
    }
    P_vap = props['P_vap']
    rho_l = props['rho_l']
    h_fg  = props['h_fg']

    L = L_over_D_ox * D_ox
    A = np.pi * (D_ox/2)**2
    steps = 120

    x = np.linspace(0, L, steps)
    P = np.linspace(P_up, P_back, steps)

    alpha = np.zeros(steps)
    alpha[0] = 1e-12
    k = 1.0

    for i in range(1, steps):
        dP_dx = (P[i] - P[i-1]) / (x[i] - x[i-1])
        if P[i-1] > P_vap:
            dalpha_dx = 0.0
        else:
            dalpha_dx = k * 2 * alpha[i-1] * (1 - alpha[i-1]) * (-dP_dx) / (rho_l * h_fg)
        alpha[i] = np.clip(alpha[i-1] + dalpha_dx * (x[i] - x[i-1]), 0.0, 0.999)

    alpha_exit = alpha[-1]
    rho_exit = (1 - alpha_exit) * rho_l + alpha_exit * props['rho_v']
    v_exit = np.sqrt(2 * (P_up - P_back) / (rho_l * (1 - alpha_exit)))
    return N_ox * rho_exit * A * v_exit

# =============================================================================
# Transient ODE system
# =============================================================================
def engine_transient(t, y):
    mdot_ox, mdot_f, mdot_comb, Pc = y

    P_feed_ox = P_tank
    P_feed_f  = P_tank

    deltaP_ox = max(P_feed_ox - Pc, 0.0)
    deltaP_f  = max(P_feed_f  - Pc, 0.0)

    # Oxidizer: La Luna flashing flow
    if deltaP_ox > 1e4:
        mdot_ox_actual = n2o_massflow_la_luna(P_feed_ox, Pc)
    else:
        mdot_ox_actual = 0.0

    # Fuel: standard incompressible orifice
    mdot_f_actual = CdA_f * np.sqrt(2 * rho_f * deltaP_f)

    # Inertance terms
    dmdot_ox_dt = (1/L_inert_ox) * (P_feed_ox - Pc - (mdot_ox**2) / (2 * CdA_ox**2 * PropsSI('D', 'T', T_n2o, 'Q', 0, fluid_n2o)))
    dmdot_f_dt  = (1/L_inert_f ) * (P_feed_f  - Pc - (mdot_f **2) / (2 * rho_f * CdA_f**2))

    # Smooth ramp using current flow state
    dmdot_ox_dt = 0.9 * dmdot_ox_dt + 0.1 * (mdot_ox_actual - mdot_ox) / 0.001
    dmdot_f_dt  = 0.9 * dmdot_f_dt  + 0.1 * (mdot_f_actual  - mdot_f ) / 0.001

    # Combustion lag
    mdot_total_in = mdot_ox + mdot_f
    dmdot_comb_dt = (mdot_total_in - mdot_comb) / tau_comb

    # Chamber pressure
    mdot_out = Pc * A_t / c_star
    dPc_dt = (gamma * R_gas * T_c / V_c) * (mdot_comb - mdot_out)

    return [dmdot_ox_dt, dmdot_f_dt, dmdot_comb_dt, dPc_dt]

# =============================================================================
# Solve transient
# =============================================================================
t_span = (0.0, 0.6)
t_eval = np.linspace(0, 0.6, 800)
y0 = [0.0, 0.0, 0.0, 1e5]   # start from ambient

print("Running transient startup simulation with La Luna N₂O flashing model...")
sol = solve_ivp(engine_transient, t_span, y0, method='LSODA',
                t_eval=t_eval, rtol=1e-6, atol=1e-9, max_step=0.001)

# =============================================================================
# Results
# =============================================================================
Pc_bar = sol.y[3] / 1e5
mdot_ox = sol.y[0]
mdot_f  = sol.y[1]
mdot_total = mdot_ox + mdot_f

print(f"\nFinal steady state:")
print(f"   Pc      = {Pc_bar[-1]:.1f} bar")
print(f"   mdot_ox = {mdot_ox[-1]:.2f} kg/s")
print(f"   mdot_f  = {mdot_f[-1]:.2f} kg/s")
print(f"   O/F     = {mdot_ox[-1]/mdot_f[-1]:.2f}")
print(f"   Thrust  ≈ {(mdot_total[-1] * c_star * 9.81 / 1000):.1f} kN")

# =============================================================================
# Plot
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(sol.t, Pc_bar, 'b-', linewidth=2.5, label='Chamber Pressure')
ax1.set_ylabel('Pc [bar]')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_title('N₂O Pressure-Fed Engine Startup Transient\n(FML Flashing Model + CoolProp)')

ax2.plot(sol.t, mdot_ox, 'r-', label='mdot N₂O (La Luna FML)')
ax2.plot(sol.t, mdot_f,  'g-', label='mdot Fuel (incompressible)')
ax2.plot(sol.t, mdot_total, 'k--', linewidth=2, label='mdot total')
ax2.set_ylabel('Mass flow [kg/s]')
ax2.set_xlabel('Time [s]')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()