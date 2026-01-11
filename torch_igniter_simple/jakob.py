import numpy as np
import pandas as pd
from pathlib import Path
from CoolProp.CoolProp import PropsSI

# ============================================================
# User settings (must match Modelica clamp ranges!)
# ============================================================
FLUID_N2O = "REFPROP::N2O"
FLUID_N2  = "REFPROP::N2"

# N2O gas table axes (T,p)
T_gas_min = 200.0
T_gas_max = 300.0
p_gas_min = 1e5
p_gas_max = 7e6

nT_gas = 80
nP_gas = 80

# EOS table axes (T,rho) for N2O gas
nRho = 80  # number of density points for p(T,rho) gas EOS

# Saturation table (1D) for N2O
T_sat_min = 182.83
T_sat_max = 309.02
nT_sat = 200

# N2 (pressurant) 1D T-axis
T_N2_min = 200.0
T_N2_max = 400.0
nT_N2    = 200

# Safety offsets
PSAT_REL_EPS = 1e-3  # keep p just below psat for gas props

# Output folder (Resources)
OUT_DIR = Path("Resources")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Helpers to write CombiTable CSVs
# ============================================================
def write_combitable2d_csv(filename, table_name, x1_vals, x2_vals, Z):
    """
    Create Modelica CombiTable2Ds style CSV:
    first row: [table_name, x2_1, x2_2, ...]
    first col of each row: [x1_i, Z_i1, Z_i2, ...]
    """
    x1_vals = np.asarray(x1_vals)
    x2_vals = np.asarray(x2_vals)
    Z = np.asarray(Z)

    header = np.concatenate(([table_name], x2_vals))
    rows = []
    for i, x1 in enumerate(x1_vals):
        rows.append(np.concatenate(([x1], Z[i, :])))

    data = np.vstack([header, *rows])
    df = pd.DataFrame(data)
    df.to_csv(OUT_DIR / filename, header=False, index=False)


def write_1d_csv(filename, table_name, x_vals, cols_dict):
    """
    Generic CombiTable1Ds style CSV with:
    first row: [table_name, col1name, col2name, ...]
    then rows: [x, col1, col2, ...]
    """
    colnames = list(cols_dict.keys())
    header = [table_name] + colnames
    data = np.column_stack([x_vals] + [cols_dict[k] for k in colnames])
    df = pd.DataFrame(np.vstack([header, data]))
    df.to_csv(OUT_DIR / filename, header=False, index=False)


# ============================================================
# 1) N2O Saturation 1D table:
#    psat, rhoL, rhoV, hL, hV, uL, uV, cpL, cpV, muL, muV, kL, kV
# ============================================================
print("Generating N2O saturation tables ...")

T_sat = np.linspace(T_sat_min, T_sat_max, nT_sat)

psat = np.zeros_like(T_sat)
rhoL = np.zeros_like(T_sat)
rhoV = np.zeros_like(T_sat)
hL   = np.zeros_like(T_sat)
hV   = np.zeros_like(T_sat)
uL   = np.zeros_like(T_sat)
uV   = np.zeros_like(T_sat)
cpL  = np.zeros_like(T_sat)
cpV  = np.zeros_like(T_sat)
muL  = np.zeros_like(T_sat)
muV  = np.zeros_like(T_sat)
kL   = np.zeros_like(T_sat)
kV   = np.zeros_like(T_sat)

for i, T in enumerate(T_sat):
    # Saturation pressure
    ps = PropsSI("P", "T", T, "Q", 0, FLUID_N2O)
    psat[i] = ps

    # Saturated liquid / vapor densities
    rhoL[i] = PropsSI("Dmass", "T", T, "Q", 0, FLUID_N2O)
    rhoV[i] = PropsSI("Dmass", "T", T, "Q", 1, FLUID_N2O)

    # Saturated liquid / vapor enthalpies
    hL[i]   = PropsSI("Hmass", "T", T, "Q", 0, FLUID_N2O)
    hV[i]   = PropsSI("Hmass", "T", T, "Q", 1, FLUID_N2O)

    # Saturated liquid / vapor internal energies
    uL[i]   = PropsSI("Umass", "T", T, "Q", 0, FLUID_N2O)
    uV[i]   = PropsSI("Umass", "T", T, "Q", 1, FLUID_N2O)

    # Saturated cp (liquid + vapor)
    cpL[i]  = PropsSI("Cpmass", "T", T, "Q", 0, FLUID_N2O)
    cpV[i]  = PropsSI("Cpmass", "T", T, "Q", 1, FLUID_N2O)

    # Saturated viscosities
    muL[i]  = PropsSI("VISCOSITY", "T", T, "Q", 0, FLUID_N2O)
    muV[i]  = PropsSI("VISCOSITY", "T", T, "Q", 1, FLUID_N2O)

    # Saturated thermal conductivities
    kL[i]   = PropsSI("CONDUCTIVITY", "T", T, "Q", 0, FLUID_N2O)
    kV[i]   = PropsSI("CONDUCTIVITY", "T", T, "Q", 1, FLUID_N2O)

write_1d_csv(
    "N2O_sat_1D.csv",
    "N2O_sat_1D",
    T_sat,
    {
        "psat": psat,
        "rhoL": rhoL,
        "rhoV": rhoV,
        "hL": hL,
        "hV": hV,
        "uL": uL,
        "uV": uV,
        "cpL": cpL,
        "cpV": cpV,
        "muL": muL,
        "muV": muV,
        "kL":  kL,
        "kV":  kV
    }
)

# ============================================================
# 2) N2O Gas property tables on (T,p):
#    cp, h, u, rho, Z, mu, k, a
# ============================================================
print("Generating N2O gas (T,p) tables ...")

T_gas  = np.linspace(T_gas_min, T_gas_max, nT_gas)
p_axis = np.linspace(p_gas_min, p_gas_max, nP_gas)

cp_gas   = np.zeros((nT_gas, nP_gas))
h_gas    = np.zeros((nT_gas, nP_gas))
u_gas    = np.zeros((nT_gas, nP_gas))
rho_gas  = np.zeros((nT_gas, nP_gas))
Z_gas    = np.zeros((nT_gas, nP_gas))
mu_gas   = np.zeros((nT_gas, nP_gas))
k_gas    = np.zeros((nT_gas, nP_gas))
a_gas    = np.zeros((nT_gas, nP_gas))

# Specific gas constant from CoolProp for N2O
R_spec_N2O = PropsSI("GAS_CONSTANT", FLUID_N2O) / PropsSI("MOLAR_MASS", FLUID_N2O)

for i, T in enumerate(T_gas):
    # Saturation pressure at this T
    ps = PropsSI("P", "T", T, "Q", 0, FLUID_N2O)
    # Don't exceed psat for gas properties
    p_max_T = min(p_gas_max, ps * (1.0 - PSAT_REL_EPS))

    for j, p in enumerate(p_axis):
        p_use = min(p, p_max_T)

        # Real-gas properties via CoolProp (gas side)
        cp_gas[i, j]   = PropsSI("Cpmass",       "T", T, "P", p_use, FLUID_N2O)
        h_gas[i, j]    = PropsSI("Hmass",        "T", T, "P", p_use, FLUID_N2O)
        u_gas[i, j]    = PropsSI("Umass",        "T", T, "P", p_use, FLUID_N2O)
        rho_gas[i, j]  = PropsSI("Dmass",        "T", T, "P", p_use, FLUID_N2O)
        mu_gas[i, j]   = PropsSI("VISCOSITY",    "T", T, "P", p_use, FLUID_N2O)
        k_gas[i, j]    = PropsSI("CONDUCTIVITY", "T", T, "P", p_use, FLUID_N2O)
        a_gas[i, j]    = PropsSI("SPEED_OF_SOUND","T",T, "P", p_use, FLUID_N2O)

        # Compressibility factor Z = p / (rho * R * T)
        Z_gas[i, j]    = p_use / (rho_gas[i, j] * R_spec_N2O * T)

write_combitable2d_csv("N2O_cp_gas_2D.csv",   "N2O_cp_gas_2D",   T_gas, p_axis, cp_gas)
write_combitable2d_csv("N2O_h_gas_2D.csv",    "N2O_h_gas_2D",    T_gas, p_axis, h_gas)
write_combitable2d_csv("N2O_u_gas_2D.csv",    "N2O_u_gas_2D",    T_gas, p_axis, u_gas)
write_combitable2d_csv("N2O_rho_gas_2D.csv",  "N2O_rho_gas_2D",  T_gas, p_axis, rho_gas)
write_combitable2d_csv("N2O_Z_gas_2D.csv",    "N2O_Z_gas_2D",    T_gas, p_axis, Z_gas)
write_combitable2d_csv("N2O_mu_gas_2D.csv",   "N2O_mu_gas_2D",   T_gas, p_axis, mu_gas)
write_combitable2d_csv("N2O_k_gas_2D.csv",    "N2O_k_gas_2D",    T_gas, p_axis, k_gas)
write_combitable2d_csv("N2O_a_gas_2D.csv",    "N2O_a_gas_2D",    T_gas, p_axis, a_gas)

# ============================================================
# 3) N2O gas EOS table p(T, rho) on (T, rho) + u, h, Z
#    (gas-only domain, as in your original script)
# ============================================================
print("Generating N2O gas EOS (T,rho) tables ...")

# Determine global rho range from safe gas domain
rho_min_list = []
rho_max_list = []

for T in T_gas:
    ps = PropsSI("P", "T", T, "Q", 0, FLUID_N2O)
    p_max_T = min(p_gas_max, ps * (1.0 - PSAT_REL_EPS))

    rho_min_list.append(PropsSI("Dmass", "T", T, "P", p_gas_min, FLUID_N2O))
    rho_max_list.append(PropsSI("Dmass", "T", T, "P", p_max_T,   FLUID_N2O))

rho_min_gas = float(np.min(rho_min_list))
rho_max_gas = float(np.max(rho_max_list))

# Density axis (log spacing is robust for wide ranges)
rho_axis = np.geomspace(max(1e-3, rho_min_gas), rho_max_gas, nRho)

p_Trho   = np.zeros((nT_gas, nRho))
u_Trho   = np.zeros((nT_gas, nRho))
h_Trho   = np.zeros((nT_gas, nRho))
Z_Trho   = np.zeros((nT_gas, nRho))

for i, T in enumerate(T_gas):
    ps = PropsSI("P", "T", T, "Q", 0, FLUID_N2O)
    p_max_T = min(p_gas_max, ps * (1.0 - PSAT_REL_EPS))
    rho_max_T = PropsSI("Dmass", "T", T, "P", p_max_T, FLUID_N2O)

    for j, rho in enumerate(rho_axis):
        # stay inside gas single-phase region
        rho_use = min(rho, rho_max_T * (1.0 - 1e-6))
        p_use = PropsSI("P", "T", T, "Dmass", rho_use, FLUID_N2O)

        # Clamp to just below psat if we numerisch an der Grenze sind
        if p_use >= ps:
            p_use = ps * (1.0 - PSAT_REL_EPS)

        p_Trho[i, j] = p_use
        u_Trho[i, j] = PropsSI("Umass", "T", T, "Dmass", rho_use, FLUID_N2O)
        h_Trho[i, j] = PropsSI("Hmass", "T", T, "Dmass", rho_use, FLUID_N2O)
        Z_Trho[i, j] = p_use / (rho_use * R_spec_N2O * T)

write_combitable2d_csv(
    "N2O_p_Trho_gas_2D.csv",
    "N2O_p_Trho_gas_2D",
    T_gas,
    rho_axis,
    p_Trho
)
write_combitable2d_csv(
    "N2O_u_Trho_gas_2D.csv",
    "N2O_u_Trho_gas_2D",
    T_gas,
    rho_axis,
    u_Trho
)
write_combitable2d_csv(
    "N2O_h_Trho_gas_2D.csv",
    "N2O_h_Trho_gas_2D",
    T_gas,
    rho_axis,
    h_Trho
)
write_combitable2d_csv(
    "N2O_Z_Trho_gas_2D.csv",
    "N2O_Z_Trho_gas_2D",
    T_gas,
    rho_axis,
    Z_Trho
)

# ============================================================
# 4) N2 (pressurant) 1D table vs T:
#    cp, h, u, mu, k  (ideal-gas-ish, p_ref = 1 bar)
# ============================================================
print("Generating N2 (pressurant) 1D tables ...")

T_N2 = np.linspace(T_N2_min, T_N2_max, nT_N2)
p_ref_N2 = 1e5  # reference pressure for property evaluation

cp_N2 = np.zeros_like(T_N2)
h_N2  = np.zeros_like(T_N2)
u_N2  = np.zeros_like(T_N2)
mu_N2 = np.zeros_like(T_N2)
k_N2  = np.zeros_like(T_N2)

for i, T in enumerate(T_N2):
    cp_N2[i] = PropsSI("Cpmass",       "T", T, "P", p_ref_N2, FLUID_N2)
    h_N2[i]  = PropsSI("Hmass",        "T", T, "P", p_ref_N2, FLUID_N2)
    u_N2[i]  = PropsSI("Umass",        "T", T, "P", p_ref_N2, FLUID_N2)
    mu_N2[i] = PropsSI("VISCOSITY",    "T", T, "P", p_ref_N2, FLUID_N2)
    k_N2[i]  = PropsSI("CONDUCTIVITY", "T", T, "P", p_ref_N2, FLUID_N2)

write_1d_csv(
    "N2_thermo_1D.csv",
    "N2_thermo_1D",
    T_N2,
    {
        "cp": cp_N2,
        "h":  h_N2,
        "u":  u_N2,
        "mu": mu_N2,
        "k":  k_N2
    }
)

# ============================================================
# Summary
# ============================================================
print("Tables written to:", OUT_DIR.resolve())
print("N2O gas (T,p) axis:", (T_gas_min, T_gas_max), (p_gas_min, p_gas_max))
print("N2O gas EOS (T,rho) axis:", (T_gas_min, T_gas_max), (rho_axis[0], rho_axis[-1]))
print("N2O saturation T range:", (T_sat_min, T_sat_max))
print("N2 T range:", (T_N2_min, T_N2_max))