import math
import numpy as np
import matplotlib.pyplot as plt

# Try to import CoolProp
try:
    import CoolProp.CoolProp as CP

    USE_COOLPROP = True
except ImportError:
    USE_COOLPROP = False
    print("Warning: CoolProp not found. Using fallback hardcoded properties.")


# --- Property Wrapper ---
class N2OProperties:
    def __init__(self):
        self.fluid = "NitrousOxide"
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
                # Likely failed to converge or out of bounds
                return 100.0, 100000.0, 0.5

        # Fallback linear approx
        return 100.0, 150000.0, 0.1

    def get_sat_densities(self, P_sat):
        """Returns (rho_liq, rho_gas) at saturation pressure."""
        if USE_COOLPROP:
            rl = CP.PropsSI('D', 'P', P_sat, 'Q', 0, self.fluid)
            rg = CP.PropsSI('D', 'P', P_sat, 'Q', 1, self.fluid)
            return rl, rg
        return self.fallback_rhol, self.fallback_rhog


props = N2OProperties()


# --- Flow Models ---

def calc_spi(C_d, Area, rho_in, P_in, P_out):
    dp = P_in - P_out
    if dp <= 0: return 0.0
    return C_d * Area * math.sqrt(2 * rho_in * dp)


def calc_hem(C_d, Area, h_in, s_in, P_out):
    rho_out, h_out, _ = props.get_isentropic_flash(P_out, s_in)
    dh = h_in - h_out
    if dh <= 0: return 0.0
    velocity = math.sqrt(2 * dh)
    return C_d * Area * rho_out * velocity


def calculate_void_fraction(x, rho_l, rho_g):
    """Homogeneous void fraction model."""
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    # Slip ratio S = 1 for homogeneous
    gamma = rho_g / rho_l
    alpha = 1.0 / (1.0 + ((1.0 - x) / x) * gamma)
    return alpha


def run_comparison_sweep():
    # --- Input Parameters ---
    d_orf = 0.8             # mm, Orifice diameter
    C_d = 0.65              # Discharge coefficient
    P_0_bar = 100.0         # bar, Upstream pressure
    T_0_K = 293.15          # K, Upstream temperature

    # Derived
    Area = math.pi * ((d_orf / 1000) / 2) ** 2
    P_0 = P_0_bar * 1e5

    # Inlet State
    P_sat = props.get_saturation_pressure(T_0_K)
    rho_in, h_in, s_in = props.get_inlet_state(P_0, T_0_K)
    rho_l_sat, rho_g_sat = props.get_sat_densities(P_sat)

    print(f"--- Simulation Setup ---")
    print(f"Fluid: N2O | T_0: {T_0_K} K | P_0: {P_0_bar} bar")
    print(f"P_sat at Inlet Temp: {P_sat / 1e5:.2f} bar")
    print(f"Orifice: {d_orf} mm | C_d: {C_d}")
    if P_0 < P_sat:
        print("Warning: Tank pressure < Saturation pressure (Two-phase in tank).")

    # Sweep Setup
    # Sweep from Upstream Pressure down to 1 bar
    pressure_steps = 100
    P_back_range = np.linspace(P_0 * 0.99, 1e5, pressure_steps)

    # Data Storage
    results = {
        "P_back": [],
        "SPI": [],
        "HEM": [],
        "Dyer": [],
        "FML": []
    }

    print("\nRunning pressure sweep...")

    for P_out in P_back_range:
        results["P_back"].append(P_out / 1e5)  # Store in bar

        # 1. Base Models
        m_spi = calc_spi(C_d, Area, rho_in, P_0, P_out)
        m_hem = calc_hem(C_d, Area, h_in, s_in, P_out)

        results["SPI"].append(m_spi)
        results["HEM"].append(m_hem)

        # 2. Dyer Model (Standard NHNE)
        # Weighting factor k based on pressure ratio
        if P_out >= P_sat:
            m_dyer = m_spi  # Pure liquid region
        else:
            try:
                k = math.sqrt((P_0 - P_out) / (P_sat - P_out))
                # m = (m_spi + k*m_hem) / (1+k)
                m_dyer = (m_spi + k * m_hem) / (1.0 + k)
            except (ValueError, ZeroDivisionError):
                m_dyer = m_spi
        results["Dyer"].append(m_dyer)

        # 3. FML Model (Void Fraction Weighting)
        # Weighting factor based on alpha (Void Fraction)
        if P_out >= P_sat:
            m_fml = m_spi
        else:
            # Get quality x from HEM flash
            _, _, x_flash = props.get_isentropic_flash(P_out, s_in)
            alpha = calculate_void_fraction(x_flash, rho_l_sat, rho_g_sat)

            # FML Weighting: transitions from SPI to HEM as void fraction increases
            # w_hem = alpha, w_spi = 1 - alpha
            m_fml = (1.0 - alpha) * m_spi + (alpha) * m_hem

        results["FML"].append(m_fml)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    # Convert mass flow to g/s for easier reading
    p_axis = results["P_back"]
    spi_data = np.array(results["SPI"]) * 1000
    hem_data = np.array(results["HEM"]) * 1000
    dyer_data = np.array(results["Dyer"]) * 1000
    fml_data = np.array(results["FML"]) * 1000

    plt.plot(p_axis, spi_data, '--', label='SPI (Incompressible)', color='gray', alpha=0.6)
    plt.plot(p_axis, hem_data, '-.', label='HEM (Equilibrium)', color='green', alpha=0.6)
    plt.plot(p_axis, dyer_data, '-', label='Dyer (NHNE)', color='blue', linewidth=2)
    plt.plot(p_axis, fml_data, '-', label='FML (Void Fraction)', color='red', linewidth=2)

    # Highlight Peaks (Choke Points)
    # The flow naturally chokes at the maximum of the curve.

    def annotate_peak(p_arr, m_arr, name, col):
        idx_max = np.argmax(m_arr)
        p_crit = p_arr[idx_max]
        m_crit = m_arr[idx_max]
        plt.plot(p_crit, m_crit, 'o', color=col)
        plt.annotate(f"{name}\n{m_crit:.1f} g/s @ {p_crit:.1f} bar",
                     (p_crit, m_crit), textcoords="offset points", xytext=(0, 10), ha='center', color=col)

    annotate_peak(p_axis, hem_data, "HEM Crit", "green")
    annotate_peak(p_axis, dyer_data, "Dyer Crit", "blue")
    annotate_peak(p_axis, fml_data, "FML Crit", "red")

    plt.title(f'N2O Injector Model Comparison (P_0={P_0_bar} bar)')
    plt.xlabel('Back Pressure (bar)')
    plt.ylabel('Mass Flow Rate (g/s)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.gca().invert_xaxis()  # Standard rocket practice: High P (left) -> Low P (right)

    print("Plot generated.")
    plt.show()


if __name__ == "__main__":
    run_comparison_sweep()