import numpy as np
from matplotlib import pyplot as plt

from fluid_lib.incompressible_orifice import inc_mass_flow

# --- Orifice Input ---
num_elements = 3    # Number of injector elements
num_orf = 3         # Orf per element
d_orf = 0.7e-3     # m, Orifice diameter

A = np.pi * (d_orf / 2) ** 2 * num_orf * num_elements   # Total orifice area
Cd = 0.6

# --- Fluid Input ---
P_0 = 45e5  # Pa
P_1 = 25e5  # Pa
T_0 = 290  # K





# ------------------- Sweep back pressure -------------------
P_back_list = np.linspace(1e5, P_0 * 0.99, 50)  # from 1 bar to almost P_0
m_dot_list = []
regime_list = []
P_throat_list = []
quality_list = []

print("Running back-pressure sweep for incompressible model...")
for P_back in P_back_list:
    m_dot = inc_mass_flow(
        fluid='REFPROP::Ethanol',
        P_0=P_0,
        T_0=T_0,
        A_orf=A,
        C_d=Cd,
        P_1=P_back
    )
    m_dot_list.append(m_dot)

m_dot_arr = np.array(m_dot_list)
P_back_bar = P_back_list / 1e5

# ------------------- Plotting -------------------
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax1 = plt.subplots(figsize=(10, 6))

# Mass flow rate
color = 'tab:blue'
ax1.set_xlabel('Back Pressure $P_1$ [bar]', fontsize=12)
ax1.set_ylabel('Mass Flow Rate [g/s]', color=color, fontsize=12)
ax1.plot(P_back_bar, m_dot_arr * 1000, color=color, linewidth=2.5, label='Predicted $\dot{m}$')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, color="grey", alpha=0.6)
ax1.axvline(x=P_1 * 1e-5, color='red', linestyle='--', linewidth=1.5,
            label=f'Back-pressure point ≈ {P_1 * 1e-5:.1f} bar')

ax1.set_title('Injector Mass Flow vs. Back Pressure\n'
              f'P_0 = {P_0 / 1e5:.0f} bar, T_0 = {T_0 - 273.15:.0f}°C, '
              f'd = {d_orf * 1000:.2f} mm, C_d = {Cd:.2f}', fontsize=14, pad=15)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(lines1, labels1, loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()
