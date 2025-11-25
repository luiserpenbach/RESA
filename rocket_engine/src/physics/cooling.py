import numpy as np
import CoolProp.CoolProp as CP
from scipy.optimize import root_scalar
from typing import Literal, Dict, Any

# Import geometry
from geometry.cooling import CoolingChannelGeometry


class RegenCoolingSolver:
    def __init__(self,
                 fluid_name: str,
                 geometry: CoolingChannelGeometry,
                 wall_conductivity: float = 15):
        """
        1D Regenerative Cooling Solver.
        """
        self.fluid = fluid_name
        self.geo = geometry
        self.k_wall = wall_conductivity
        self.num_stations = len(geometry.x_contour)

    def solve(self,
              mdot_coolant_total: float,
              pin_coolant: float,
              tin_coolant: float,
              T_gas_recovery: np.ndarray,
              h_gas: np.ndarray,
              mode: Literal['co-flow', 'counter-flow'] = 'counter-flow') -> Dict[str, np.ndarray]:
        """
        Marches down the channel solving for thermal equilibrium.

        Args:
            mode: 'co-flow' (Inlet at Injector) or 'counter-flow' (Inlet at Nozzle Exit)
        """

        # --- PRE-PROCESSING: Direction Handling ---
        # We define working arrays. If counter-flow, we flip them so index 0 is always Coolant Inlet.
        if mode == 'counter-flow':
            # Flip Geometry
            x_work = np.flip(self.geo.x_contour)
            r_work = np.flip(self.geo.radius_contour)
            w_work = np.flip(self.geo.channel_width)
            h_work = np.flip(self.geo.channel_height)
            rib_work = np.flip(self.geo.rib_width)
            tw_work = np.flip(self.geo.wall_thickness)

            # Flip Gas Boundary Conditions (must match physical location of coolant)
            Tg_work = np.flip(T_gas_recovery)
            hg_work = np.flip(h_gas)
        else:
            # Co-Flow: Use as is
            x_work = self.geo.x_contour
            r_work = self.geo.radius_contour
            w_work = self.geo.channel_width
            h_work = self.geo.channel_height
            rib_work = self.geo.rib_width
            tw_work = self.geo.wall_thickness

            Tg_work = T_gas_recovery
            hg_work = h_gas

        # 1. Initialize Result Arrays
        N = self.num_stations
        res = {
            "T_coolant": np.zeros(N),
            "P_coolant": np.zeros(N),
            "T_wall_hot": np.zeros(N),
            "T_wall_cold": np.zeros(N),
            "q_flux": np.zeros(N),
            "velocity": np.zeros(N),
            "quality": np.zeros(N)
        }

        # 2. Setup Initial State (At Coolant Inlet)
        current_P = pin_coolant
        current_H = CP.PropsSI('H', 'T', tin_coolant, 'P', current_P, self.fluid)

        # Calculate N channels based on Inlet Geometry (which is now correctly index 0)
        R_inlet = r_work[0] + tw_work[0]
        circumference = 2 * np.pi * R_inlet

        # Explicit scalar conversion to avoid TypeError
        width_inlet = float(w_work[0])
        rib_inlet = float(rib_work[0])

        num_channels = int(np.floor(circumference / (width_inlet + rib_inlet)))
        mdot_channel = mdot_coolant_total / num_channels

        print(f"--- Starting Cooling Solve ({mode}): {num_channels} ch, mdot/ch={mdot_channel:.4f} kg/s ---")

        # 3. Marching Loop
        for i in range(N):
            # A. Get Local Geometry (from working arrays)
            width = w_work[i]
            height = h_work[i]
            t_wall_local = tw_work[i]

            D_h = (2 * width * height) / (width + height)
            Area_flow = width * height

            # B. Update Fluid Properties
            try:
                if current_P < 1000: raise ValueError("Pressure too low")

                # Fluid property lookups
                T_bulk = CP.PropsSI('T', 'H', current_H, 'P', current_P, self.fluid)
                rho = CP.PropsSI('D', 'H', current_H, 'P', current_P, self.fluid)
                visc = CP.PropsSI('V', 'H', current_H, 'P', current_P, self.fluid)
                cond = CP.PropsSI('L', 'H', current_H, 'P', current_P, self.fluid)
                cp = CP.PropsSI('Cpmass', 'H', current_H, 'P', current_P, self.fluid)
                prandtl = (cp * visc) / cond
            except ValueError:
                print(f"CoolProp Error at Station {i}. P={current_P:.1f}, H={current_H:.1f}")
                break

            velocity = mdot_channel / (rho * Area_flow)
            Re = (rho * velocity * D_h) / visc

            # C. Solve Heat Transfer Balance
            Tg = Tg_work[i]
            hg = hg_work[i]

            # Helper for H_coolant side coeff
            def get_hc(T_wall_c):
                # Dittus-Boelter
                Nu = 0.023 * (Re ** 0.8) * (prandtl ** 0.4)

                h_base = (cond / D_h) * Nu

                # Fin Effect (Simplified)
                # Assumes heating on 3 sides (Bottom + 2 Walls) if ribs are efficient
                # or just Bottom if ribs are insulating.
                # Standard approximation for regen:
                perimeter_wetted = width + 2 * height
                h_eff = h_base * (perimeter_wetted / width)

                return h_eff

            # Residual: Q_in - Q_out = 0
            def heat_balance_residual(Twg):
                q_flux_in = hg * (Tg - Twg)
                Twc = Twg - (q_flux_in * t_wall_local / self.k_wall)

                hc = get_hc(Twc)
                q_flux_out = hc * (Twc - T_bulk)
                return q_flux_in - q_flux_out

            try:
                # Bracket: [T_coolant, T_gas]
                sol = root_scalar(heat_balance_residual, bracket=[T_bulk, Tg], method='brentq', xtol=1e-2)
                Twg = sol.root
            except ValueError:
                Twg = T_bulk + 50

                # D. Save Step Results
            q_flux = hg * (Tg - Twg)
            Twc = Twg - (q_flux * t_wall_local / self.k_wall)

            # E. Advance to Next Station
            if i < N - 1:
                # Distance to next station
                dx = abs(x_work[i + 1] - x_work[i])

                # 1. Enthalpy Rise (Heat Addition)
                perimeter_chamber = 2 * np.pi * r_work[i]
                area_segment = perimeter_chamber * dx
                q_total_segment = q_flux * area_segment / num_channels

                delta_H = q_total_segment / mdot_channel
                current_H += delta_H

                # 2. Pressure Drop (Friction)
                f = self._friction_factor(Re, self.geo.roughness, D_h)
                dp_friction = f * (dx / D_h) * (0.5 * rho * velocity ** 2)
                current_P -= dp_friction

            # Record
            res["T_coolant"][i] = T_bulk
            res["P_coolant"][i] = current_P
            res["T_wall_hot"][i] = Twg
            res["T_wall_cold"][i] = Twc
            res["q_flux"][i] = q_flux
            res["velocity"][i] = velocity
            try:
                res["quality"][i] = CP.PropsSI('Q', 'H', current_H, 'P', current_P, self.fluid)
            except:
                res["quality"][i] = -1

        # --- POST-PROCESSING: Flip Back if Counter-Flow ---
        # We want the output arrays to align with self.geo.x_contour (Injector -> Nozzle)
        if mode == 'counter-flow':
            for key in res:
                res[key] = np.flip(res[key])

        return res

    def _friction_factor(self, Re, roughness, Dh):
        if Re < 2000: return 64.0 / Re
        rel_rough = roughness / Dh
        # Avoid log(0)
        if rel_rough <= 0 and Re > 0:
            return 0.3164 * Re ** -0.25  # Blasius for smooth pipes

        return (-1.8 * np.log10((rel_rough / 3.7) ** 1.11 + 6.9 / Re)) ** -2