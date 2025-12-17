import CoolProp.CoolProp as CP
import numpy as np
import matplotlib.pyplot as plt
import math


class CavitatingVenturi:
    def __init__(self, fluid, T_in_C, P_in_bar, m_dot_target, diffuser_angle=7, Cd=0.96):
        """
        Initialize the Venturi Sizing Tool.

        Parameters:
        -----------
        fluid : str
            CoolProp fluid string (e.g., 'Water', 'Acetone', 'Propane')
        T_in_C : float
            Inlet Temperature [Celsius]
        P_in_bar : float
            Inlet Pressure [bar absolute]
        m_dot_target : float
            Target Mass Flow Rate at Choked conditions [kg/s]
        diffuser_angle : float
            Total cone angle of the diffuser [degrees].
            Standard is 7-15 deg. Nozzle is >30 deg.
        Cd : float
            Discharge Coefficient (Default 0.96)
        """
        self.fluid = fluid
        self.T_in = T_in_C + 273.15
        self.P_in = P_in_bar * 1e5
        self.m_dot_target = m_dot_target
        self.angle = diffuser_angle
        self.Cd = Cd

        # Initialize properties
        self.rho_in = None
        self.P_sat = None
        self.geom = {}  # Will hold sizing results

        # Run initialization calculations
        self._get_fluid_properties()
        self._calc_FL_factor()
        self._size_throat()

    def _get_fluid_properties(self):
        """Calculates density and vapor pressure."""
        try:
            self.rho_in = CP.PropsSI('D', 'P', self.P_in, 'T', self.T_in, self.fluid)
            self.P_sat = CP.PropsSI('P', 'T', self.T_in, 'Q', 0, self.fluid)

            # Validation
            if self.P_in <= self.P_sat:
                raise ValueError("Inlet Pressure is below Vapor Pressure! Fluid is already boiling.")

        except Exception as e:
            print(f"Fluid Property Error: {e}")
            raise

    def _calc_FL_factor(self):
        """Estimates Pressure Recovery Factor (FL) based on diffuser angle."""
        # Empirical approximation based on Idelchik data
        alpha = self.angle
        if alpha < 5:
            k_loss = 0.12
        elif alpha <= 20:
            slope = (0.25 - 0.10) / (20 - 6)
            k_loss = 0.10 + slope * (alpha - 6)
        elif alpha <= 60:
            slope = (0.90 - 0.25) / (60 - 20)
            k_loss = 0.25 + slope * (alpha - 20)
        else:
            k_loss = 1.0

        self.FL = math.sqrt(min(k_loss, 1.0))

    def _size_throat(self):
        """Calculates the required throat diameter."""
        # 1. Max available pressure drop at throat (Internal)
        delta_p_cav = self.P_in - self.P_sat

        # 2. Area Calculation (Bernoulli)
        # m_dot = Cd * A * sqrt(2 * rho * dp)
        self.area_throat = self.m_dot_target / (self.Cd * math.sqrt(2 * self.rho_in * delta_p_cav))

        # 3. Diameter Calculation
        self.dia_throat = math.sqrt((4 * self.area_throat) / math.pi)

        # Store geometry
        self.geom['diameter_mm'] = self.dia_throat * 1000.0
        self.geom['area_mm2'] = self.area_throat * 1e6

        # Calculate Critical Pressure Drop observable at the Valve (P1-P2)
        # dP_valve_crit = FL^2 * (P1 - Psat)
        self.dp_valve_critical = (self.FL ** 2) * delta_p_cav

    def print_design_report(self):
        """Prints a summary of the design to console."""
        print("=" * 50)
        print(f"VENTURI DESIGN REPORT: {self.fluid}")
        print("=" * 50)
        print(f"Inputs:")
        print(f"  P_in: {self.P_in / 1e5:.2f} bar | T_in: {self.T_in - 273.15:.2f} °C")
        print(f"  Target Flow: {self.m_dot_target} kg/s")
        print(f"  Diffuser Angle: {self.angle}°")
        print("-" * 50)
        print(f"Fluid Properties:")
        print(f"  Vapor Pressure: {self.P_sat / 1e5:.4f} bar")
        print(f"  Density: {self.rho_in:.2f} kg/m3")
        print("-" * 50)
        print(f"Sized Geometry:")
        print(f"  Throat Diameter: **{self.geom['diameter_mm']:.4f} mm**")
        print(f"  Throat Area: {self.geom['area_mm2']:.2f} mm2")
        print("-" * 50)
        print(f"Performance Characteristics:")
        print(f"  Recovery Factor (FL): {self.FL:.3f} (Est.)")
        print(f"  Critical dP (P1-P2):  {self.dp_valve_critical / 1e5:.3f} bar")
        print(f"  (At this dP, the valve chokes and flow flatlines)")
        print("=" * 50)

    def plot_performance(self):
        """Generates the performance curve."""

        # Sweep Outlet Pressure from P_in down to 0
        P_out_sweep = np.linspace(self.P_in, 0, 100)

        mass_flows = []
        valve_dps = []
        is_choked = []

        max_throat_dp = self.P_in - self.P_sat

        for P_out in P_out_sweep:
            dp_valve = self.P_in - P_out

            # Determine Internal Throat conditions
            # If valve dP is low, we are in normal flow
            # If valve dP > Critical, we are Choked

            if dp_valve < self.dp_valve_critical:
                # Unchoked: Relate valve dP to throat dP via FL
                dp_throat_effective = dp_valve / (self.FL ** 2)
                choked = False
            else:
                # Choked: Throat dP is clamped at max (P_in - P_sat)
                dp_throat_effective = max_throat_dp
                choked = True

            # Calculate Flow
            if dp_throat_effective <= 0:
                m_dot = 0
            else:
                m_dot = self.Cd * self.area_throat * math.sqrt(2 * self.rho_in * dp_throat_effective)

            mass_flows.append(m_dot)
            valve_dps.append(dp_valve / 1e5)
            is_choked.append(choked)

        # PLOTTING
        plt.figure(figsize=(10, 6))
        x = np.array(valve_dps)
        y = np.array(mass_flows)
        mask = np.array(is_choked)

        plt.plot(x[~mask], y[~mask], 'b-', linewidth=2, label='Normal Hydraulic Flow')
        plt.plot(x[mask], y[mask], 'r--', linewidth=2, label='Cavitating (Choked) Flow')

        # Add Reference Lines
        crit_bar = self.dp_valve_critical / 1e5
        plt.axvline(x=crit_bar, color='k', linestyle=':', alpha=0.6)
        plt.text(crit_bar + 0.2, self.m_dot_target * 0.9,
                 f'Critical $\Delta P$\n{crit_bar:.2f} bar', fontsize=10)

        plt.title(f'Venturi Performance Curve: {self.fluid} (FL={self.FL:.2f})', fontsize=14)
        plt.xlabel('Measured Pressure Drop ($P_{in} - P_{out}$) [bar]', fontsize=12)
        plt.ylabel('Mass Flow Rate [kg/s]', fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(loc=4)
        plt.tight_layout()
        plt.show()


# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # --- USER INPUTS ---
    design = CavitatingVenturi(
        fluid='REFPROP::Ethanol',
        T_in_C=20.0,
        P_in_bar=100.0,
        m_dot_target=0.2,  # We want to limit flow
        diffuser_angle=40.0  # Standard efficient Venturi
    )

    # 1. Print Sizing Numbers
    design.print_design_report()

    # 2. Show Simulation Plot
    design.plot_performance()