import CoolProp.CoolProp as CP
import numpy as np

### FLUID INITIALIZATION
fluid = "REFPROP::N2"          # Nitrogen using REFPROP backend (most accurate)
T_0 = 300.0                    # K
p_0 = 100.0                     # bar (upstream)
p_1 = 75.0                     # bar (downstream)
orf_d = 8.5                    # mm → orifice diameter
orf_cd = 0.7                   # discharge coefficient



# Convert pressures to Pa for CoolProp
P_0_Pa = p_0 * 1e5
P_1_Pa = p_1 * 1e5

# Thermodynamic properties at stagnation state (0)
rho_0 = CP.PropsSI("D", "T", T_0, "P", P_0_Pa, fluid)   # kg/m³
cp    = CP.PropsSI("CPMASS", "T", T_0, "P", P_0_Pa, fluid)
cv    = CP.PropsSI("CVMASS", "T", T_0, "P", P_0_Pa, fluid)
k     = cp / cv                                          # isentropic exponent γ
M     = CP.PropsSI("M", "T", T_0, "P", P_0_Pa, fluid)    # kg/mol (not needed here)
print(f"Isentropic exponent γ        : {k:.4f}")
print(f"Upstream density ρ₀           : {rho_0:.3f} kg/m³")

# Critical pressure ratio
pr_crit = (2 / (k + 1)) ** (k / (k - 1))
pr      = p_1 / p_0
print(f"Pressure ratio p₁/p₀          : {pr:.3f}")
print(f"Critical pressure ratio       : {pr_crit:.3f}")

# Area of one orifice
A = np.pi / 4 * (orf_d * 1e-3) ** 2                     # m²

if pr < pr_crit:
    print(">>> Choked Flow (critical) <<<")
    # Choked (sonic) mass flow – ideal gas formula (very accurate for N2 here)
    m_dot = orf_cd * A * rho_0 * np.sqrt( k * (2/(k+1))**((k+1)/(k-1)) * (P_0_Pa / rho_0) )
else:
    print(">>> Subcritical flow <<<")
    m_dot = orf_cd * A * np.sqrt( 2 * rho_0 * P_0_Pa * (k/(k-1)) *
              ( (p_1/p_0)**(2/k) - (p_1/p_0)**((k+1)/k) ) )

# Volumetric flow converted to Normal conditions (0°C, 1.01325 bar)
Q_actual_m3h = m_dot / rho_0 * 3600                     # actual m³/h at upstream T,p
T_ref = 293.15                                          # 20 °C (common industrial normal condition in many places)
P_ref = 1.01325                                         # bar
Q_Nm3h = Q_actual_m3h * (T_ref / T_0) * (p_0 / P_ref)   # Nm³/h

print(f"Mass flow rate (1 orifice)      : {m_dot:.6f} kg/s")


# For Injector Configuration with multiple orifices
num_inj_elements = 3
num_ports_per_injector = 5
total_orifices = num_inj_elements * num_ports_per_injector

print("\n--- Results per single orifice ---")
print(f"Mass flow rate (1 orifice)      : {m_dot:.6f} kg/s")
print(f"                                = {m_dot*3600:.2f} kg/h")
print(f"Normal volumetric flow (1 orifice): {Q_Nm3h:.4f} Nm³/h")
print(f"                                = {Q_Nm3h*1000/60:.3f} NL/min")

print("\n--- Total for the complete injector system ---")
print(f"Total orifices                  : {total_orifices}")
print(f"Total mass flow                 : {m_dot*total_orifices:.4f} kg/s")
print(f"                                = {m_dot*total_orifices*3600:.2f} kg/h")
print(f"Total Normal flow               : {Q_Nm3h*total_orifices:.2f} Nm³/h")
print(f"                                = {Q_Nm3h*total_orifices*1000/60:.2f} NL/min")
