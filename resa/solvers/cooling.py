"""Regenerative Cooling Solver for RESA."""
import numpy as np
import CoolProp.CoolProp as CP
from scipy.optimize import root_scalar
from typing import Literal, Dict, Any

from resa.core.results import CoolingResult, CoolingChannelGeometry


class RegenCoolingSolver:
    """
    1D Regenerative Cooling Analysis Solver.

    Performs thermal-hydraulic analysis of cooling channels
    using a marching method along the channel length.
    """

    def __init__(self, wall_conductivity: float = 15.0):
        """
        Initialize cooling solver.

        Args:
            wall_conductivity: Thermal conductivity of wall material [W/m-K]
                              Default 15 W/m-K for copper alloy
        """
        self.k_wall = wall_conductivity

    def solve(
        self,
        mdot_coolant: float,
        p_in: float,
        t_in: float,
        geometry: CoolingChannelGeometry,
        T_gas: np.ndarray,
        h_gas: np.ndarray,
        mode: Literal['co-flow', 'counter-flow'] = 'counter-flow',
        fluid_name: str = 'Ethanol'
    ) -> CoolingResult:
        """
        Solve regenerative cooling along the channel.

        Args:
            mdot_coolant: Total coolant mass flow rate [kg/s]
            p_in: Coolant inlet pressure [Pa]
            t_in: Coolant inlet temperature [K]
            geometry: Cooling channel geometry
            T_gas: Recovery temperature array [K]
            h_gas: Gas-side heat transfer coefficient array [W/m2-K]
            mode: Flow direction ('co-flow' or 'counter-flow')
            fluid_name: CoolProp fluid name

        Returns:
            CoolingResult with thermal analysis data
        """
        N = len(geometry.x)

        # Direction handling - flip arrays for counter-flow
        if mode == 'counter-flow':
            x_work = np.flip(geometry.x)
            r_work = np.flip(geometry.y)
            w_work = np.flip(geometry.channel_width)
            h_work = np.flip(geometry.channel_height)
            rib_work = np.flip(geometry.rib_width)
            tw_work = np.flip(geometry.wall_thickness)
            Tg_work = np.flip(T_gas)
            hg_work = np.flip(h_gas)
        else:
            x_work = geometry.x
            r_work = geometry.y
            w_work = geometry.channel_width
            h_work = geometry.channel_height
            rib_work = geometry.rib_width
            tw_work = geometry.wall_thickness
            Tg_work = T_gas
            hg_work = h_gas

        # Initialize result arrays
        T_coolant = np.zeros(N)
        P_coolant = np.zeros(N)
        T_wall_hot = np.zeros(N)
        T_wall_cold = np.zeros(N)
        q_flux = np.zeros(N)
        velocity = np.zeros(N)

        # Initial state
        current_P = p_in
        current_H = CP.PropsSI('H', 'T', t_in, 'P', current_P, fluid_name)

        # Calculate number of channels
        R_inlet = r_work[0] + tw_work[0]
        circumference = 2 * np.pi * R_inlet
        width_inlet = float(w_work[0])
        rib_inlet = float(rib_work[0])
        num_channels = int(np.floor(circumference / (width_inlet + rib_inlet)))
        mdot_channel = mdot_coolant / num_channels

        # Marching loop
        for i in range(N):
            width = w_work[i]
            height = h_work[i]
            t_wall_local = tw_work[i]

            D_h = (2 * width * height) / (width + height)
            Area_flow = width * height

            # Fluid properties
            try:
                if current_P < 1000:
                    raise ValueError("Pressure too low")

                T_bulk = CP.PropsSI('T', 'H', current_H, 'P', current_P, fluid_name)
                rho = CP.PropsSI('D', 'H', current_H, 'P', current_P, fluid_name)
                visc = CP.PropsSI('V', 'H', current_H, 'P', current_P, fluid_name)
                cond = CP.PropsSI('L', 'H', current_H, 'P', current_P, fluid_name)
                cp = CP.PropsSI('Cpmass', 'H', current_H, 'P', current_P, fluid_name)
                prandtl = (cp * visc) / cond
            except ValueError:
                break

            vel = mdot_channel / (rho * Area_flow)
            Re = (rho * vel * D_h) / visc

            # Heat transfer balance
            Tg = Tg_work[i]
            hg = hg_work[i]

            def get_hc(T_wall_c):
                Nu = 0.023 * (Re ** 0.8) * (prandtl ** 0.4)
                h_base = (cond / D_h) * Nu
                perimeter_wetted = width + 2 * height
                return h_base * (perimeter_wetted / width)

            def heat_balance_residual(Twg):
                q_flux_in = hg * (Tg - Twg)
                Twc = Twg - (q_flux_in * t_wall_local / self.k_wall)
                hc = get_hc(Twc)
                q_flux_out = hc * (Twc - T_bulk)
                return q_flux_in - q_flux_out

            try:
                sol = root_scalar(
                    heat_balance_residual,
                    bracket=[T_bulk, Tg],
                    method='brentq',
                    xtol=1e-2
                )
                Twg = sol.root
            except ValueError:
                Twg = T_bulk + 50

            q = hg * (Tg - Twg)
            Twc = Twg - (q * t_wall_local / self.k_wall)

            # Advance to next station
            if i < N - 1:
                dx_axial = abs(x_work[i + 1] - x_work[i])
                perimeter_chamber = 2 * np.pi * r_work[i]
                area_segment = perimeter_chamber * dx_axial
                q_total_segment = q * area_segment / num_channels

                delta_H = q_total_segment / mdot_channel
                current_H += delta_H

                # Pressure drop
                f = self._friction_factor(Re, geometry.roughness, D_h)
                dp_friction = f * (dx_axial / D_h) * (0.5 * rho * vel ** 2)
                current_P -= dp_friction

            # Record
            T_coolant[i] = T_bulk
            P_coolant[i] = current_P
            T_wall_hot[i] = Twg
            T_wall_cold[i] = Twc
            q_flux[i] = q
            velocity[i] = vel

        # Flip back for counter-flow
        if mode == 'counter-flow':
            T_coolant = np.flip(T_coolant)
            P_coolant = np.flip(P_coolant)
            T_wall_hot = np.flip(T_wall_hot)
            T_wall_cold = np.flip(T_wall_cold)
            q_flux = np.flip(q_flux)
            velocity = np.flip(velocity)

        # Calculate pressure drop
        pressure_drop = (p_in - P_coolant[-1]) / 1e5  # Convert to bar

        return CoolingResult(
            T_coolant=T_coolant,
            P_coolant=P_coolant,
            T_wall_hot=T_wall_hot,
            T_wall_cold=T_wall_cold,
            q_flux=q_flux,
            velocity=velocity,
            max_wall_temp=float(np.max(T_wall_hot)),
            max_heat_flux=float(np.max(q_flux)),
            pressure_drop=pressure_drop,
            outlet_temp=float(T_coolant[-1] if mode == 'co-flow' else T_coolant[0])
        )

    def _friction_factor(self, Re: float, roughness: float, Dh: float) -> float:
        """Calculate Darcy friction factor using Goudar-Sonnad approximation."""
        if Re < 2000:
            return 64.0 / Re

        epsilon = roughness / Dh
        a = 2.0 / np.log(10)
        b = epsilon / 3.7
        d = (np.log(10) * Re) / 5.02

        s = b * d + np.log(d)
        q = s ** (s / (s + 1))
        g = b * d + np.log(d / q)
        z = np.log(q / g)

        delta_la = (z * g) / (g + 1)
        delta_cfa = delta_la * (1 + (z / 2) / ((g + 1) ** 2 + (z / 3) * (2 * g - 1)))

        inv_sqrt_f = a * (np.log(d / q) + delta_cfa)
        return (1.0 / inv_sqrt_f) ** 2
