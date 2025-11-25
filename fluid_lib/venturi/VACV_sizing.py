import CoolProp.CoolProp as CP
import numpy as np
import matplotlib.pyplot as plt
import math


class VariableAreaVenturi:
    def __init__(self, fluid, P_tank_bar, T_in_C, max_flow_kg_s):
        self.fluid = fluid
        self.P_in = P_tank_bar * 1e5
        self.T_in = T_in_C + 273.15

        # Fluid Props
        self.rho = CP.PropsSI('D', 'P', self.P_in, 'T', self.T_in, fluid)
        self.Psat = CP.PropsSI('P', 'T', self.T_in, 'Q', 0, fluid)

        # Sizing the max area for the target flow at 100% open
        # Assuming choked condition for sizing
        dp_cav = self.P_in - self.Psat
        Cd = 0.95
        self.A_max = max_flow_kg_s / (Cd * math.sqrt(2 * self.rho * dp_cav))

    def get_flow(self, percent_open, P_chamber_bar):
        """
        Calculates flow for a given valve opening and chamber pressure.
        Returns both the Cavitating Flow (Theoretical) and Normal Flow.
        """
        if percent_open <= 0: return 0.0

        P_out = P_chamber_bar * 1e5

        # Linear Area Characteristic for this simulation
        # Area = Max_Area * %Open
        A_current = self.A_max * (percent_open / 100.0)
        Cd = 0.95

        # 1. Calculate Cavitating Flow limit (Independent of P_out)
        dp_cav = self.P_in - self.Psat
        m_dot_cav = Cd * A_current * math.sqrt(2 * self.rho * dp_cav)

        # 2. Calculate Hydraulic Flow (Dependent on P_out)
        dp_hydraulic = self.P_in - P_out
        if dp_hydraulic <= 0:
            m_dot_hyd = 0
        else:
            m_dot_hyd = Cd * A_current * math.sqrt(2 * self.rho * dp_hydraulic)

        # The actual flow is the MINIMUM of the two
        # If Hydraulic < Cavitating -> We are NOT choked (Normal valve behavior)
        # If Hydraulic > Cavitating -> We ARE choked (Flow limited by cavitation)

        actual_flow = min(m_dot_hyd, m_dot_cav)
        is_cavitating = (m_dot_cav < m_dot_hyd)

        return actual_flow, is_cavitating


def simulate_throttling_scenario():
    # --- SCENARIO: LANDING BURN ---
    # We are throttling a LOX valve from 100% down to 20%
    # As we throttle down, Chamber Pressure (P_c) drops roughly linearly with flow.

    valve = VariableAreaVenturi('Oxygen', P_tank_bar=40.0, T_in_C=-180, max_flow_kg_s=5.0)

    # Sweep valve position from 100% open down to 10% open
    valve_positions = np.linspace(100, 10, 50)

    # Storage
    flows_ideal = []
    flows_real = []
    chamber_pressures = []

    print(f"Simulating Throttling for {valve.fluid}...")

    # We iterate to find equilibrium (simple convergence)
    # Relationship: P_chamber approx proportional to Flow
    # P_c = k * m_dot
    # k approx 30 bar / 5 kg/s = 6 bar per kg/s
    K_combustor = 30e5 / 5.0

    for pos in valve_positions:

        # Initial guess for flow
        m_dot = 0

        # Simple loop to resolve the coupling between Flow and Chamber Pressure
        # (This simulates what happens in a real engine)
        P_c_current = 30.0  # start guess

        for _ in range(5):  # Convergence loop
            m_dot, is_cav = valve.get_flow(pos, P_c_current)

            # Physics: Chamber pressure results from mass flow
            P_c_pascal = m_dot * K_combustor
            P_c_current = P_c_pascal / 1e5

        flows_real.append(m_dot)
        chamber_pressures.append(P_c_current)

        # Calculate what flow would be if P_c didn't affect it (Pure Cavitation)
        # This represents the "Ideal Control" line
        flows_ideal.append(valve.get_flow(pos, 0)[0])

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))

    # 1. Plot Mass Flow vs Valve Position
    plt.plot(valve_positions, flows_real, 'b-o', label='Actual Engine Flow')
    plt.plot(valve_positions, flows_ideal, 'r--', label='Ideal Linear Reference')

    plt.title('Throttling Linearity: Cavitating Venturi', fontsize=14)
    plt.xlabel('Valve Opening (%)', fontsize=12)
    plt.ylabel('Mass Flow Rate (kg/s)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()

    # Add secondary axis for Chamber Pressure
    ax2 = plt.gca().twinx()
    ax2.plot(valve_positions, chamber_pressures, 'g:', alpha=0.5)
    ax2.set_ylabel('Chamber Pressure (bar)', color='g', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='g')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate_throttling_scenario()