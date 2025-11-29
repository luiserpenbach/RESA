import numpy as np
from scipy.integrate import solve_ivp
from src.components.feed_system import FluidLine, InjectorComponent


class TransientSimulation:
    def __init__(self, engine_result, config):
        self.res = engine_result
        self.cfg = config

        # Engine Physics Props
        self.Vc = self._estimate_chamber_volume()
        self.At = np.pi * (self.res.dt_mm / 2000) ** 2

        # Combustion Props (Simplified)
        self.c_star = self.res.cooling_data.get('c_star', 1350.0)  # Should be in result
        self.gamma = 1.25  # Ideally from CEA result
        self.R_gas = 280.0  # J/kgK (Approx)
        self.Tc = self.res.T_gas_recovery[0]  # Chamber Temp

        # Setup Feed Components
        # Assume some defaults or add to Config
        self.line_ox = FluidLine(length=3.5, diameter=10*1e-3, roughness=10e-5)
        self.line_fu = FluidLine(length=1.5, diameter=10*1e-3, roughness=10e-5)

        # Injectors (Sized from Design)
        # Recover geometric area from design result
        # Assuming we stored the injector design in 'result' or re-calculate
        self.inj_ox = InjectorComponent(15, 2*1e-3, fluid_type="n2o")
        self.inj_fu = InjectorComponent(9, 0.35*1e-3, fluid_type="incompressible")

    def _estimate_chamber_volume(self):
        # Vc = L* * At
        At = np.pi * (self.res.dt_mm / 2000) ** 2
        return (self.cfg.L_star / 1000) * At

    def run(self, t_end=0.5):
        # State Vector: [mdot_ox, mdot_f, Pc]
        # Ignoring combustion lag state for simplicity, solving Pc directly

        y0 = [0.0, 0.0, 1e5]  # Start at 1 bar
        t_span = (0, t_end)

        sol = solve_ivp(self._derivative, t_span, y0, method='LSODA', rtol=1e-5)
        return sol

    def _derivative(self, t, y):
        mdot_ox_line, mdot_f_line, Pc = y

        # 1. Tank Conditions
        P_tank_ox = 60 * 1e5
        P_tank_fu = 100 * 1e5  # Assumed higher than Pc

        # 2. Injector Flow (Demand)
        # Using current Pc as backpressure
        mdot_ox_inj = self.inj_ox.get_mass_flow(P_tank_ox, Pc, 285, 750)
        mdot_f_inj = self.inj_fu.get_mass_flow(P_tank_fu, Pc, 298, 789)

        # 3. Line Dynamics (Inertance)
        # d(mdot)/dt = (1/L) * (P_tank - P_inj - dP_friction)
        # Note: P_inj is approximately Pc + dP_injector
        # Simplified: Treat the whole line + injector as one impedance
        # d(mdot)/dt = (P_tank - Pc - dP_line - dP_inj_loss) / Inertance

        # Friction Loss
        dp_line_ox = self.line_ox.calculate_dp(mdot_ox_line, 750, 1e-5)
        dp_line_fu = self.line_fu.calculate_dp(mdot_f_line, 789, 1e-3)

        # Pressure at Injector Inlet
        P_inj_in_ox = P_tank_ox - dp_line_ox
        P_inj_in_fu = P_tank_fu - dp_line_fu

        # Actual flow through injector based on Line Momentum
        # This is the coupling.
        # Easier approach: d(mdot)/dt = (P_tank - P_inj_in) / Inertance ??
        # Standard approach:
        # Inertance * dmdot/dt = P_tank - P_chamber - (dP_friction + dP_injector)

        # Re-calc Injector Drop based on CURRENT Line Velocity mass flow
        # dP = (mdot / CdA)^2 / (2 rho) ... inverse of flow equation
        dp_inj_ox = (mdot_ox_line / (self.inj_ox.area_total * self.inj_ox.cd)) ** 2 / (2 * 750)
        dp_inj_fu = (mdot_f_line / (self.inj_fu.area_total * self.inj_fu.cd)) ** 2 / (2 * 789)

        d_mdot_ox = (P_tank_ox - Pc - dp_line_ox - dp_inj_ox) / self.line_ox.inertance
        d_mdot_fu = (P_tank_fu - Pc - dp_line_fu - dp_inj_fu) / self.line_fu.inertance

        # 4. Chamber Dynamics
        # dPc/dt = (gamma * R * Tc / Vc) * (mdot_in - mdot_out)
        mdot_in = mdot_ox_line + mdot_f_line
        mdot_out = Pc * self.At / self.c_star

        d_Pc = (self.gamma * self.R_gas * self.Tc / self.Vc) * (mdot_in - mdot_out)

        return [d_mdot_ox, d_mdot_fu, d_Pc]