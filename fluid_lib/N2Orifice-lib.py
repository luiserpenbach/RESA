import math
import numpy as np
import sys

# Try to import CoolProp
try:
    import CoolProp.CoolProp as CP

    USE_COOLPROP = True
except ImportError:
    USE_COOLPROP = False
    print("Warning: CoolProp not found. Using fallback hardcoded properties.")


class N2OProperties:
    """
    Internal helper to handle thermodynamic property lookups.
    """

    def __init__(self):
        self.fluid = "REFPROP::NitrousOxide"
        # Pre-calculate saturation properties at 20C for fallback
        self.fallback_Psat = 50.2e5
        self.fallback_rhol = 775.0
        self.fallback_rhog = 100.0

    def get_saturation_pressure(self, T_K):
        if USE_COOLPROP:
            return CP.PropsSI('P', 'T', T_K, 'Q', 0, self.fluid)
        return self.fallback_Psat

    def get_inlet_state(self, P_in, T_in):
        """Returns (rho, h, s) at inlet conditions."""
        if USE_COOLPROP:
            rho = CP.PropsSI('D', 'P', P_in, 'T', T_in, self.fluid)
            h = CP.PropsSI('H', 'P', P_in, 'T', T_in, self.fluid)
            s = CP.PropsSI('S', 'P', P_in, 'T', T_in, self.fluid)
            return rho, h, s
        return self.fallback_rhol, 200000.0, 1000.0

    def get_isentropic_flash(self, P_out, s_in):
        """Returns (rho_out, h_out, x_out) for isentropic expansion to P_out."""
        if USE_COOLPROP:
            try:
                rho = CP.PropsSI('D', 'P', P_out, 'S', s_in, self.fluid)
                h = CP.PropsSI('H', 'P', P_out, 'S', s_in, self.fluid)
                # CoolProp Q returns -1 for single phase liquid, check bounds
                x = CP.PropsSI('Q', 'P', P_out, 'S', s_in, self.fluid)
                if x < 0: x = 0.0
                if x > 1: x = 1.0
                return rho, h, x
            except:
                return 100.0, 100000.0, 0.5

        return 100.0, 150000.0, 0.1

    def get_sat_densities(self, P_sat):
        """Returns (rho_liq, rho_gas) at saturation pressure."""
        if USE_COOLPROP:
            rl = CP.PropsSI('D', 'P', P_sat, 'Q', 0, self.fluid)
            rg = CP.PropsSI('D', 'P', P_sat, 'Q', 1, self.fluid)
            return rl, rg
        return self.fallback_rhol, self.fallback_rhog


class N2Orifice:
    """
    Calculates N2O mass flow through an injector orifice using various
    two-phase flow models (SPI, HEM, Dyer, FML).
    """

    def __init__(self, diameter_mm, cd, p_tank_bar, t_tank_k):
        self.props = N2OProperties()

        # Geometry
        self.d = diameter_mm / 1000.0
        self.area = math.pi * (self.d / 2) ** 2
        self.cd = cd

        # Tank State
        self.p_tank = p_tank_bar * 1e5
        self.t_tank = t_tank_k

        # Pre-calculate Inlet Properties (Optimization)
        self.p_sat = self.props.get_saturation_pressure(self.t_tank)
        self.rho_in, self.h_in, self.s_in = self.props.get_inlet_state(self.p_tank, self.t_tank)
        self.rho_l_sat, self.rho_g_sat = self.props.get_sat_densities(self.p_sat)

    def _calc_void_fraction(self, x):
        """Homogeneous void fraction model."""
        if x <= 0: return 0.0
        if x >= 1: return 1.0
        gamma = self.rho_g_sat / self.rho_l_sat
        alpha = 1.0 / (1.0 + ((1.0 - x) / x) * gamma)
        return alpha

    def spi_flow(self, p_back):
        """Single Phase Incompressible flow at specific back pressure."""
        dp = self.p_tank - p_back
        if dp <= 0: return 0.0
        return self.cd * self.area * math.sqrt(2 * self.rho_in * dp)

    def hem_flow(self, p_back):
        """Homogeneous Equilibrium Model flow at specific back pressure."""
        rho_out, h_out, _ = self.props.get_isentropic_flash(p_back, self.s_in)
        dh = self.h_in - h_out
        if dh <= 0: return 0.0
        velocity = math.sqrt(2 * dh)
        return self.cd * self.area * rho_out * velocity

    def dyer_flow(self, p_back):
        """Dyer (NHNE) flow at specific back pressure (uncorrected for choking)."""
        m_spi = self.spi_flow(p_back)
        m_hem = self.hem_flow(p_back)

        if p_back >= self.p_sat:
            return m_spi

        try:
            k = math.sqrt((self.p_tank - p_back) / (self.p_sat - p_back))
            # Weighted Average
            return (m_spi + k * m_hem) / (1.0 + k)
        except (ValueError, ZeroDivisionError):
            return m_spi

    def fml_flow(self, p_back):
        """FML flow at specific back pressure (uncorrected for choking)."""
        m_spi = self.spi_flow(p_back)
        m_hem = self.hem_flow(p_back)

        if p_back >= self.p_sat:
            return m_spi

        _, _, x_flash = self.props.get_isentropic_flash(p_back, self.s_in)
        alpha = self._calc_void_fraction(x_flash)

        # Weighting based on Void Fraction
        return (1.0 - alpha) * m_spi + (alpha) * m_hem

    def get_critical_flow(self, model='DYER', p_min_bar=1.0, steps=100):
        """
        Iterates to find the maximum (choked) mass flow rate for the selected model.
        Returns: (max_mass_flow_kg_s, critical_pressure_bar)
        """
        p_min = p_min_bar * 1e5
        p_range = np.linspace(self.p_tank * 0.99, p_min, steps)

        max_flow = 0.0
        crit_p = p_min

        for p in p_range:
            if model.upper() == 'DYER':
                m = self.dyer_flow(p)
            elif model.upper() == 'FML':
                m = self.fml_flow(p)
            elif model.upper() == 'HEM':
                m = self.hem_flow(p)
            else:
                m = self.spi_flow(p)  # SPI doesn't really choke, but included for completeness

            if m > max_flow:
                max_flow = m
                crit_p = p
            else:
                # If flow decreases, we passed the choke point (for converging nozzles)
                # Optimization: we could break here, but continuing is safer for complex curves
                pass

        return max_flow, crit_p / 1e5

    def get_curve(self, model='DYER', p_min_bar=1.0, steps=100):
        """Returns arrays (pressure_bar, mass_flow_kg_s) for the full curve."""
        p_min = p_min_bar * 1e5
        p_arr = np.linspace(self.p_tank * 0.99, p_min, steps)
        m_arr = []

        for p in p_arr:
            if model.upper() == 'DYER':
                m_arr.append(self.dyer_flow(p))
            elif model.upper() == 'FML':
                m_arr.append(self.fml_flow(p))
            elif model.upper() == 'HEM':
                m_arr.append(self.hem_flow(p))
            else:
                m_arr.append(self.spi_flow(p))

        return np.array(p_arr) / 1e5, np.array(m_arr)


# --- Example Usage ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1. Setup
    a = np.pi/4*2.5**2*15
    d_mono = np.sqrt(a/np.pi*4)
    injector = N2Orifice(diameter_mm=d_mono, cd=0.65, p_tank_bar=45.0, t_tank_k=293.15)

    print(f"Injector initialized: {injector.d * 1000}mm @ {injector.p_tank / 1e5} bar")

    # 2. Calculate Critical Flows (Choked values)
    mdot_dyer, p_crit_dyer = injector.get_critical_flow('DYER')
    mdot_fml, p_crit_fml = injector.get_critical_flow('FML')

    print(f"Dyer Critical Flow: {mdot_dyer:.4f} kg/s @ {p_crit_dyer:.1f} bar")
    print(f"FML  Critical Flow: {mdot_fml:.4f} kg/s @ {p_crit_fml:.1f} bar")

    # 3. Plot Curves
    p_axis, m_dyer_curve = injector.get_curve('DYER')
    _, m_fml_curve = injector.get_curve('FML')
    _, m_spi_curve = injector.get_curve('SPI')
    _, m_hem_curve = injector.get_curve('HEM')

    plt.figure(figsize=(12, 6))
    plt.plot(p_axis, m_spi_curve * 1000, '--', color='gray', alpha=0.5, label='SPI')
    plt.plot(p_axis, m_hem_curve * 1000, '-.', color='green', alpha=0.5, label='HEM')
    plt.plot(p_axis, m_dyer_curve * 1000, '-', color='blue', linewidth=2, label='Dyer')
    plt.plot(p_axis, m_fml_curve * 1000, '-', color='red', linewidth=2, label='FML')

    # Mark Choke Points
    plt.plot(p_crit_dyer, mdot_dyer * 1000, 'o', color='blue')
    plt.plot(p_crit_fml, mdot_fml * 1000, 'o', color='red')

    plt.gca().invert_xaxis()
    plt.xlabel("Back Pressure (bar)")
    plt.ylabel("Mass Flow (g/s)")
    plt.title("N2O Orifice Flow Models")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()