import numpy as np
from CoolProp.CoolProp import PropsSI
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def inc_mass_flow(
        P_0,  # Tank pressure [Pa]
        T_0,  # Tank temperature [K]
        A_orf,  # orifice area [m²]
        C_d=0.9,  # Discharge coefficient (typical 0.6–0.9 for sharp orifices)
        P_1=None  # Back pressure [Pa], if None → assumes vacuum
):
    """
    Incompressible flow model for nitrous oxide (N2O) mass flow through an orifice.

    Parameters:
        P_0        : Stagnation pressure [Pa]
        T_0        : Stagnation temperature [K]
        A_orf  : Throat area [m²]
        C_d        : Discharge coefficient (default 0.9)
        P_1    : Ambient/back pressure [Pa]. If None, assumes choked flow into vacuum.

    Returns:
        m_dot     : Mass flow rate [kg/s]
        is_choked : Boolean indicating if flow is choked
        regime    : 'choked', 'subcritical', or 'two-phase'
    """

    fluid = 'REFPROP::Ethanol'

    # Stagnation properties
    h_0 = PropsSI('H', 'P', P_0, 'T', T_0, fluid)  # Specific enthalpy [J/kg]
    s_0 = PropsSI('S', 'P', P_0, 'T', T_0, fluid)  # Specific entropy [J/kg·K]
    rho_0 = PropsSI('D', 'P', P_0, 'T', T_0, fluid)  # Density [kg/m³]

    # Critical pressure ratio for N2O (approximately when vapor pressure is reached)
    # We'll find throat condition by isentropic expansion until minimum enthalpy (two-phase)

    # Try to find vapor quality at stagnation (usually saturated or slightly subcooled)
    try:
        Q0 = PropsSI('Q', 'P', P_0, 'T', T_0, fluid)
    except:
        Q0 = -1  # Subcooled liquid

    m_dot = C_d * A_orf * np.sqrt(2*rho_0 * (P_0-P_1))
    print(m_dot)

    return m_dot

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":

    P_0 = 100e5          # Pa
    P_1 = 60e5          # Pa
    T_0 = 290            # K
    d_orf = 2.3e-3       # m  → 8.5 mm
    A = np.pi * (d_orf / 2) ** 2
    Cd = 0.6

    # ------------------- Sweep back pressure -------------------
    P_back_list = np.linspace(1e5, P_0 * 0.99, 50)   # from 5 bar to almost P_0
    m_dot_list = []
    regime_list = []
    P_throat_list = []
    quality_list = []

    print("Running back-pressure sweep for incompressible model...")
    for P_back in P_back_list:
        m_dot = inc_mass_flow(
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
    ax1.plot(P_back_bar, m_dot_arr*1000, color=color, linewidth=2.5, label='Predicted $\dot{m}$')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, color="grey", alpha=0.6)
    ax1.axvline(x=P_1*1e-5, color='red', linestyle='--', linewidth=1.5,
                label=f'Back-pressure point ≈ {P_1*1e-5:.1f} bar')

    ax1.set_title('N2O Injector Mass Flow vs. Back Pressure\n'
                  f'P_0 = {P_0/1e5:.0f} bar, T_0 = {T_0-273.15:.0f}°C, '
                  f'd = {d_orf*1000:.1f} mm, C_d = {Cd:.2f}', fontsize=14, pad=15)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.show()

    # ------------------- Print single operating point -------------------
    P_1 = 60e5
    m_dot= inc_mass_flow(
        P_0=P_0, T_0=T_0, A_orf=A, C_d=Cd, P_1=P_1
    )

    print("\n" + "="*60)
    print(f"         SINGLE OPERATING POINT (P₁ = {P_1} bar)")
    print("="*60)
    print(f"Upstream Pressure     : {P_0/1e5:.1f} bar")
    print(f"Upstream Temperature  : {T_0-273.15:.1f} °C")
    print(f"Back Pressure         : {P_1/1e5:.1f} bar")
    print(f"Orifice Diameter      : {d_orf*1000:.2f} mm")
    print(f"Mass Flow Rate        : {m_dot*1000:.2f} g/s  ({m_dot:.4f} kg/s)")
