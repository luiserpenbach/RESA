import numpy as np
import matplotlib.pyplot as plt


def simulate_oxidizer_only_throttling():
    # --- DESIGN POINT (100% Thrust) ---
    F_thrust_target = 1.0  # Normalized Thrust
    OF_ratio_design = 4.0  # Typical N2O/Fuel ratio
    Pc_design = 25.0  # bar

    # Injector Deltas
    dPx_design = 20.0  # Oxidizer dP (your design)
    dPf_design = 20.0  # Fuel dP (Typical assumes lower dP for fuel, or matched)

    # Tank Pressures (Fixed)
    P_ox_tank = Pc_design + dPx_design  # bar
    P_fuel_tank = Pc_design + dPf_design  # bar

    # --- THROTTLING SIMULATION ---
    # We reduce Chamber Pressure (proxy for Thrust) from 25 bar down to 12.5 bar
    Pc_sweep = np.linspace(Pc_design, Pc_design * 0.5, 50)

    ox_flows = []
    fuel_flows = []
    of_ratios = []
    thrusts = []

    for Pc in Pc_sweep:
        # 1. Fuel Behavior (UNTHROTTLED)
        # Fuel flow is driven by fixed Tank P vs falling Chamber P
        dPf_actual = P_fuel_tank - Pc
        # Flow proportional to sqrt(dP)
        # m_dot_f = k * sqrt(dP)
        # Normalized: m_dot_f = sqrt(dPf_actual / dPf_design)
        m_fuel = np.sqrt(dPf_actual / dPf_design)

        # 2. Oxidizer Behavior (THROTTLED)
        # We assume we adjust the valve to achieve this Pc
        # We need to solve for required Ox flow to sustain this Pc
        # Total Flow ~ Pc (approx)
        # m_total = m_total_design * (Pc / Pc_design)

        # Total mass units at design: (6 parts Ox + 1 part Fuel) = 7 parts
        m_total_required = 7.0 * (Pc / Pc_design)

        # m_ox = m_total - m_fuel
        m_ox = m_total_required - m_fuel * 1.0  # 1.0 is fuel mass unit

        if m_ox < 0: m_ox = 0

        # Store Data
        ox_flows.append(m_ox)
        fuel_flows.append(m_fuel)

        current_OF = m_ox / (m_fuel * 1.0)
        of_ratios.append(current_OF)
        thrusts.append(Pc / Pc_design * 100)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))

    plt.plot(thrusts, of_ratios, 'r-', linewidth=3)
    plt.axhline(y=OF_ratio_design, color='k', linestyle='--', label='Design O/F')

    # Visuals
    plt.title('Oxidizer-Only Throttling', fontsize=14)
    plt.xlabel('Thrust Level (%)', fontsize=12)
    plt.ylabel('O/F Ratio', fontsize=12)
    plt.grid(True)
    plt.gca().invert_xaxis()  # Show going from 100 -> 50

    # Annotation
    final_of = of_ratios[-1]
    plt.text(55, final_of + 0.5, f'Final O/F: {final_of:.2f}', fontweight='bold')

    plt.show()


if __name__ == "__main__":
    simulate_oxidizer_only_throttling()