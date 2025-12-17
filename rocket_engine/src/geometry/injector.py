import numpy as np
from dataclasses import dataclass


@dataclass
class InjectorGeometry:
    orifice_radius: float  # r_o [mm]
    chamber_radius: float  # r_s [mm]
    inlet_radius: float  # r_p [mm]
    num_elements: int
    alpha_spray: float  # [deg]
    cd: float  # Discharge coefficient
    film_thickness: float  # [mm]


class SwirlInjectorSizer:
    def __init__(self, mdot_total: float, p_drop: float, n_elements: int, propellant_density: float):
        self.mdot = mdot_total / n_elements  # Per element
        self.dp = p_drop
        self.rho = propellant_density
        self.n_el = n_elements

    def calculate(self, target_alpha: float = 60.0, n_inlets: int = 3) -> InjectorGeometry:
        """
        Sizes a swirl injector for a target spray angle.
        """
        # 1. Determine Geometric Characteristic A (or K) based on Alpha
        # Using Lefebvre correlation for alpha vs K
        # Curve fit approximation for standard swirlers
        # Alpha = f(K). Inverting to find K from Alpha.
        # Approx: K = A / (1 + X) ...
        # Using simplified Rezende correlation: X = 0.0042 * alpha^1.27 (alpha in deg?)
        # Let's use the explicit logic from the original file snippet which was clearer.

        # Original snippet: X = 0.0042 * alpha^1.2714
        X = 0.0042 * (target_alpha ** 1.2714)
        # X = (Area_air_core / Area_orifice)

        # 2. Discharge Coefficient (Cd)
        # Relation: Cd = sqrt( (1-X)^3 / (1+X) ) (Maximum flow theory)
        cd = np.sqrt(((1 - X) ** 3) / (1 + X))

        # 3. Calculate Discharge Area (Ao)
        # mdot = Cd * Ao * sqrt(2 * rho * dp)
        # Ao = mdot / (Cd * sqrt(2 * rho * dp))
        v_theoretical = np.sqrt(2 * self.rho * self.dp)
        area_orifice = self.mdot / (cd * v_theoretical)

        r_orifice = np.sqrt(area_orifice / np.pi)

        # 4. Calculate Inlet Ports
        # Swirl strength K = Ap / (Dm * Do) ... many definitions exist.
        # Using Rezende: Cd_p = sqrt(X^3 / (2-X)) for inlet
        cd_inlet = np.sqrt((X ** 3) / (2 - X))

        # Inlet Area
        area_inlet_total = self.mdot / (cd_inlet * v_theoretical)
        r_inlet = np.sqrt(area_inlet_total / (np.pi * n_inlets))

        # 5. Swirl Chamber geometry
        # Rule of thumb: R_chamber approx 3 * R_orifice to decouple wall effects
        r_chamber = 3.0 * r_orifice

        # Film thickness (t = r_o * (1 - sqrt(X)))
        t_film = r_orifice * (1 - np.sqrt(X))

        return InjectorGeometry(
            orifice_radius=r_orifice * 1000,
            chamber_radius=r_chamber * 1000,
            inlet_radius=r_inlet * 1000,
            num_elements=self.n_el,
            alpha_spray=target_alpha,
            cd=cd,
            film_thickness=t_film * 1000
        )