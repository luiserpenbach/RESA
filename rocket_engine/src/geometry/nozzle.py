import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class NozzleGeometryData:
    """Standardized output data structure for nozzle geometry."""
    x_chamber: np.ndarray  # Cylindrical Chamber section
    y_chamber: np.ndarray
    x_chamber_ent: np.ndarray  # Entrance Arc (Cylinder -> Cone)
    y_chamber_ent: np.ndarray
    x_throat_cone: np.ndarray  # Linear Conical section
    y_throat_cone: np.ndarray
    x_throat_conv: np.ndarray  # Throat Upstream Arc (Cone -> Throat)
    y_throat_conv: np.ndarray
    x_throat_div: np.ndarray  # Throat Downstream Arc
    y_throat_div: np.ndarray
    x_bell: np.ndarray  # Bell nozzle section
    y_bell: np.ndarray

    @property
    def x_full(self) -> np.ndarray:
        return np.concatenate([
            self.x_chamber, self.x_chamber_ent, self.x_throat_cone,
            self.x_throat_conv, self.x_throat_div, self.x_bell
        ])

    @property
    def y_full(self) -> np.ndarray:
        return np.concatenate([
            self.y_chamber, self.y_chamber_ent, self.y_throat_cone,
            self.y_throat_conv, self.y_throat_div, self.y_bell
        ])


class NozzleGenerator:
    """
    Generates contour coordinates for liquid rocket engine thrust chambers.
    """

    def __init__(self, R_throat: float, expansion_ratio: float):
        self.Rt = R_throat
        self.epsilon = expansion_ratio
        self.Re = R_throat * np.sqrt(expansion_ratio)

    def generate(self,
                 contraction_ratio: float = 0.0,
                 L_star: float = 1000.0,
                 bell_fraction: float = 0.8,
                 theta_convergent: float = 45.0,
                 R_chamber: float = 0.0,
                 R_ent_factor: float = 1.0) -> NozzleGeometryData:
        """
        Args:
            R_ent_factor: Radius of the entrance arc as a fraction of Chamber Radius Rc.
                          (Standard is often 0.5 to 1.0 * Rc).
        """

        # 1. Determine Chamber Geometry
        if R_chamber > 0:
            Rc = R_chamber
            CR = (Rc / self.Rt) ** 2
        elif contraction_ratio > 0:
            CR = contraction_ratio
            Rc = self.Rt * np.sqrt(CR)
        else:
            raise ValueError("Must provide either contraction_ratio or R_chamber")

        # 2. Generate Divergent (Bell) Section
        x_bell, y_bell, x_trans_div, y_trans_div, theta_n, theta_e = self._generate_rao_bell(bell_fraction)

        # 3. Generate Convergent (Chamber) Section with Entrance Arc
        x_conv, y_conv, x_cone, y_cone, x_ent, y_ent, x_cyl, y_cyl = self._generate_chamber(
            Rc, L_star, CR, theta_convergent, R_ent_factor
        )

        return NozzleGeometryData(
            x_chamber=x_cyl,
            y_chamber=y_cyl,
            x_chamber_ent=x_ent,
            y_chamber_ent=y_ent,
            x_throat_cone=x_cone,
            y_throat_cone=y_cone,
            x_throat_conv=x_conv,
            y_throat_conv=y_conv,
            x_throat_div=x_trans_div,
            y_throat_div=y_trans_div,
            x_bell=x_bell,
            y_bell=y_bell
        )

    def _generate_rao_bell(self, length_fraction: float):
        # ... (Same as previous response) ...
        # For brevity, I will assume the previous implementation of _generate_rao_bell is used here.
        # Recopying the logic briefly for completeness of the class context:
        ar_data = [4, 5, 10, 20, 30, 40, 50, 100]
        tn_80_deg = [21.5, 23.0, 26.3, 28.8, 30.0, 31.0, 31.5, 33.5]
        te_80_deg = [14.0, 13.0, 11.0, 9.0, 8.5, 8.0, 7.5, 7.0]

        theta_n_deg = np.interp(self.epsilon, ar_data, tn_80_deg)
        theta_e_deg = np.interp(self.epsilon, ar_data, te_80_deg)
        theta_n = np.radians(theta_n_deg)
        theta_e = np.radians(theta_e_deg)
        print(f"Nozzle Entrance Angle: {theta_n_deg:.2f}")
        print(f"Nozzle Exit Angle: {theta_e_deg:.2f}")

        L_cone = (self.Re - self.Rt) / np.tan(np.radians(15))
        L_nozzle = length_fraction * L_cone

        R_arc_div = 0.382 * self.Rt
        angle_range_trans = np.linspace(0, theta_n, 20)
        x_trans = R_arc_div * np.sin(angle_range_trans)
        y_trans = self.Rt + R_arc_div * (1 - np.cos(angle_range_trans))

        Nx, Ny = x_trans[-1], y_trans[-1]
        Ex, Ey = L_nozzle, self.Re
        m1, m2 = np.tan(theta_n), np.tan(theta_e)
        C1, C2 = Ny - m1 * Nx, Ey - m2 * Ex
        Qx = (C2 - C1) / (m1 - m2)
        Qy = (m1 * C2 - m2 * C1) / (m1 - m2)

        t = np.linspace(0, 1, 100)
        x_bell = ((1 - t) ** 2) * Nx + 2 * (1 - t) * t * Qx + (t ** 2) * Ex
        y_bell = ((1 - t) ** 2) * Ny + 2 * (1 - t) * t * Qy + (t ** 2) * Ey

        return x_bell, y_bell, x_trans, y_trans, theta_n_deg, theta_e_deg

    def _generate_chamber(self, Rc: float, L_star: float, CR: float,
                          theta_conv_deg: float, R_ent_factor: float):
        """
        Generates chamber sections:
        Cylinder -> Entrance Arc -> Cone -> Throat Arc -> Throat

        CRITICAL FIX: All arrays are generated Left-to-Right (Increasing X)
        to ensure proper concatenation and cylinder sizing.
        """
        # Volume of chamber required
        Vc = L_star * (np.pi * self.Rt ** 2)

        theta = np.radians(theta_conv_deg)

        # --- 1. Throat Upstream Arc (Radius 2) ---
        # Defines the geometry near x=0 (Throat)
        R2 = 1.5 * self.Rt

        # We sweep angle alpha from Theta (left) to 0 (throat)
        # x = -R2 * sin(alpha)  -> Increases from negative to 0
        alpha2 = np.linspace(theta, 0, 20)
        x_arc_throat = -R2 * np.sin(alpha2)
        y_arc_throat = self.Rt + R2 * (1 - np.cos(alpha2))

        # Start of Throat Arc (Point B) - Leftmost point of this segment
        xB = x_arc_throat[0]
        yB = y_arc_throat[0]

        # --- 2. Entrance Arc (Radius 1) ---
        # Defines geometry coming off the cylinder
        R1 = Rc * R_ent_factor

        # Calculate tangent points
        # yA is the vertical height where the Entrance Arc meets the Cone
        # Geometry: Center of arc is at y = Rc - R1.
        yA = (Rc - R1) + R1 * np.cos(theta)

        # --- 3. Conical Section ---
        # Connects Point A (End of Ent Arc) to Point B (Start of Throat Arc)
        dy_cone = yA - yB

        if dy_cone < 0:
            # Handle overlap case (CR too small for these radii)
            print(f"Warning: Entrance/Throat radii too large for CR={CR}. Radii clipped.")
            dx_cone = 0
            x_cone = np.array([])
            y_cone = np.array([])
            xA = xB  # Cone doesn't exist, so A meets B directly
        else:
            dx_cone = dy_cone / np.tan(theta)

            # Cone goes from xA to xB
            xA = xB - dx_cone
            x_cone = np.linspace(xA, xB, 10, endpoint=False)  # exclude last point to avoid duplicate with xB

            # Linear equation for cone
            # Slope k = (yB - yA) / (xB - xA) = -tan(theta)
            # y - yA = k * (x - xA)
            slope = -np.tan(theta)
            y_cone = slope * (x_cone - xA) + yA

        # --- 2b. Generate Entrance Arc Arrays ---
        # Now we know xA (where arc ends on the right).
        # We need to find the arc center.
        # At angle theta, x = x_center + R1*sin(theta) = xA
        x_center = xA - R1 * np.sin(theta)

        # Sweep alpha from 0 (Cylinder match) to theta (Cone match)
        # This ensures X increases (Left to Right)
        alpha1 = np.linspace(0, theta, 20, endpoint=False)

        x_arc_ent = x_center + R1 * np.sin(alpha1)
        y_arc_ent = (Rc - R1) + R1 * np.cos(alpha1)

        # --- 4. Cylindrical Chamber ---
        # Concatenate everything to get the full Convergent Profile
        # Order: Ent Arc -> Cone -> Throat Arc
        x_cont = np.concatenate([x_arc_ent, x_cone, x_arc_throat])
        y_cont = np.concatenate([y_arc_ent, y_cone, y_arc_throat])

        # Integrate volume of the convergent section
        vol_conv = np.trapz(np.pi * y_cont ** 2, x_cont)

        # Determine Cylinder Length based on L*
        vol_cyl_req = Vc - vol_conv
        if vol_cyl_req < 0:
            len_cyl = 0
        else:
            len_cyl = vol_cyl_req / (np.pi * Rc ** 2)

        # The cylinder ends exactly where the Entrance Arc starts
        x_cyl_end = x_arc_ent[0] if len(x_arc_ent) > 0 else xA
        x_cyl_start = x_cyl_end - len_cyl

        x_chamber = np.linspace(x_cyl_start, x_cyl_end, 50, endpoint=False)
        y_chamber = np.full_like(x_chamber, Rc)

        return (x_arc_throat, y_arc_throat,
                x_cone, y_cone,
                x_arc_ent, y_arc_ent,
                x_chamber, y_chamber)


# --- Usage Example ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Init
    gen = NozzleGenerator(R_throat=14.35, expansion_ratio=4.0)

    # Generate
    geo = gen.generate(
        L_star=1200,
        contraction_ratio=12,
        bell_fraction=1
    )

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(geo.x_full, geo.y_full, 'k-', linewidth=2, label="Inner Wall")
    plt.plot(geo.x_full, -geo.y_full, 'k-', linewidth=2)
    plt.axis('equal')
    plt.grid(True, linestyle='--')
    plt.title("Thrust Optimized Parabolic (Rao) Nozzle")
    plt.xlabel("Axial Position [mm]")
    plt.ylabel("Radial Position [mm]")
    plt.legend()
    plt.show()