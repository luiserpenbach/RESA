from dataclasses import dataclass

import numpy as np
from rocket_engine.src.physics.fluid_dynamics import friction_factor_gs, flow_spi, flow_n2o_la_luna


@dataclass
class FluidLine:
    length: float  # m
    diameter: float  # m
    roughness: float  # m
    inertance: float = 0.0

    def __post_init__(self):
        area = np.pi * (self.diameter / 2) ** 2
        self.inertance = self.length / area

    def calculate_dp(self, mdot, rho, viscosity):
        """Calculates pressure drop for a given mass flow."""
        area = np.pi * (self.diameter / 2) ** 2
        velocity = mdot / (rho * area)
        Re = (rho * velocity * self.diameter) / viscosity

        f = friction_factor_gs(Re, self.roughness, self.diameter)
        dp = f * (self.length / self.diameter) * (0.5 * rho * velocity ** 2)
        return dp


class InjectorComponent:
    def __init__(self, n_elements, orifice_dia, cd=0.65, fluid_type="incompressible"):
        self.n = n_elements
        self.d = orifice_dia
        self.area_total = n_elements * np.pi * (orifice_dia / 2) ** 2
        self.cd = cd
        self.type = fluid_type  # 'n2o' or 'incompressible'

    def get_mass_flow(self, p_up, p_down, t_up, rho_up):
        if self.type == "n2o":
            # Use La Luna Model
            return flow_n2o_la_luna(self.area_total, p_up, p_down, t_up)
        else:
            # Use SPI Model
            dp = p_up - p_down
            return flow_spi(self.area_total, self.cd, rho_up, dp)