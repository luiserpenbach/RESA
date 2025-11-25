import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


class PintleProfiler:
    def __init__(self, throat_dia_mm, max_stroke_mm, max_flow_kg_s, fluid_density, delta_p_bar):
        """
        Generates the contour for a linear-flow pintle.
        """
        self.Rt = throat_dia_mm / 2.0
        self.L_max = max_stroke_mm

        # Calculate Target Max Area (assuming nominal Cd=0.9)
        # A = m_dot / (Cd * sqrt(2 * rho * dp))
        dp_pa = delta_p_bar * 1e5
        Cd_nominal = 0.90

        required_area_m2 = max_flow_kg_s / (Cd_nominal * math.sqrt(2 * fluid_density * dp_pa))
        self.A_max_mm2 = required_area_m2 * 1e6

        # Sanity Check: Can the throat physically pass this flow?
        self.A_throat_mm2 = math.pi * self.Rt ** 2
        if self.A_max_mm2 > self.A_throat_mm2:
            print("WARNING: Target flow requires area larger than throat!")
            print(f"Throat Area: {self.A_throat_mm2:.2f} mm2")
            print(f"Req Area:    {self.A_max_mm2:.2f} mm2")
            self.A_max_mm2 = self.A_throat_mm2 * 0.95  # Cap it at 95%

    def get_cd_at_stroke(self, percent_stroke):
        """
        Returns estimated Cd.
        Usually Cd drops slightly at low lift due to friction effects.
        We can use a simple curve fit or constant.
        """
        # Simple model: Cd varies from 0.85 (low lift) to 0.96 (full open)
        # Cd(x) = 0.85 + (0.96-0.85) * (x/100)^(0.5)
        return 0.85 + (0.11 * math.sqrt(percent_stroke / 100.0 + 0.001))

    def generate_contour(self, resolution=100):
        """
        Generates the X (axial) and Y (radius) coordinates.
        """
        x_points = np.linspace(0, self.L_max, resolution)
        r_points = []

        for x in x_points:
            # Fraction of stroke
            frac = x / self.L_max
            if frac < 0.001: frac = 0.001  # Avoid div by zero

            # 1. Determine Target Flow Area at this stroke
            # We want Linear Flow: Area_target = A_max * frac
            # BUT, we must correct for Cd variation.
            # Flow = Cd(x) * A(x) * Constant
            # To keep Flow linear, A(x) must compensate for Cd(x)

            Cd_current = self.get_cd_at_stroke(frac * 100)
            Cd_max = self.get_cd_at_stroke(100)

            # The corrected area target
            target_area = self.A_max_mm2 * frac * (Cd_max / Cd_current)

            # 2. Calculate Radius
            # A = pi * (Rt^2 - rp^2)
            # rp^2 = Rt^2 - (A / pi)

            term = self.Rt ** 2 - (target_area / math.pi)

            if term < 0:
                # This happens if we try to open wider than the throat itself
                r_pintle = 0
            else:
                r_pintle = math.sqrt(term)

            r_points.append(r_pintle)

        return x_points, np.array(r_points)

    def plot_profile(self):
        x, r = self.generate_contour()

        # Mirror for visualization (Top and Bottom profile)
        r_top = r
        r_bot = -r

        plt.figure(figsize=(12, 5))

        # 1. The Throat Wall
        plt.axhline(y=self.Rt, color='k', linewidth=3, label='Throat Wall')
        plt.axhline(y=-self.Rt, color='k', linewidth=3)

        # 2. The Pintle
        plt.fill_between(x, r_top, r_bot, color='gray', alpha=0.5, label='Pintle Body')
        plt.plot(x, r_top, 'b-', linewidth=2)
        plt.plot(x, r_bot, 'b-', linewidth=2)

        # Formatting
        plt.title('Generated Linear-Flow Pintle Profile', fontsize=14)
        plt.xlabel('Axial Stroke [mm] (Flow Direction ->)', fontsize=12)
        plt.ylabel('Radial Position [mm]', fontsize=12)
        plt.axis('equal')
        plt.grid(True)
        plt.legend(loc='upper right')

        # Annotations
        plt.text(0, 0, ' SEAT \n(Closed)', ha='center', va='center', fontweight='bold')
        plt.text(self.L_max, 0, ' MAX \nOPEN', ha='center', va='center', fontweight='bold')

        plt.tight_layout()
        plt.show()

        return pd.DataFrame({'Axial_Pos_mm': x, 'Radius_mm': r})


# --- EXECUTION ---
if __name__ == "__main__":
    # Example: LOX Valve for 5 kN engine
    profiler = PintleProfiler(
        throat_dia_mm=10.0,  # Fixed Venturi Throat
        max_stroke_mm=4.0,  # Travel distance
        max_flow_kg_s=1.5,  # Target max flow
        fluid_density=1141.0,  # LOX density
        delta_p_bar=20.0  # Tank Pressure - Vapor Pressure
    )

    df = profiler.plot_profile()

    # Check the "Nose" radius (at max stroke)
    tip_radius = df.iloc[-1]['Radius_mm']
    print(f"Pintle Radius at Seat (x=0): {df.iloc[0]['Radius_mm']:.3f} mm")
    print(f"Pintle Radius at Tip (x=max): {tip_radius:.3f} mm")

    # If Tip Radius > 0, the pintle is blunt (good for structure)
    # If Tip Radius = 0, the pintle is a sharp needle