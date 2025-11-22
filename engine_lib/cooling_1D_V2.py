import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import CoolProp.CoolProp as CP  # Use CoolProp for fluid properties


def get_fluid_props(fluid, P, T, prop):
    """
    Retrieve fluid properties using CoolProp.
    :param fluid: Fluid name (e.g., 'Oxygen', 'Hydrogen')
    :param P: Pressure in MPa
    :param T: Temperature in K
    :param prop: Property to retrieve ('D' for density, 'V' for viscosity, 'C' for Cp, 'L' for conductivity)
    :return: Property value
    """
    P_pa = P * 1e6  # Convert to Pa
    if prop == 'D':
        return CP.PropsSI('D', 'T', T, 'P', P_pa, fluid)
    elif prop == 'V':
        return CP.PropsSI('V', 'T', T, 'P', P_pa, fluid)
    elif prop == 'C':
        return CP.PropsSI('C', 'T', T, 'P', P_pa, fluid)
    elif prop == 'L':
        return CP.PropsSI('L', 'T', T, 'P', P_pa, fluid)
    else:
        raise ValueError(f"Unknown property: {prop}")


class RegenCoolingSizer:
    def __init__(self, contour_array, t_wall, fluid_coolant, P_inlet, T_inlet, mdot_coolant,
                 T_gas, P_chamber, k_wall, T_wall_max, N_channels, h_channel_initial,
                 material='Inconel', gamma_gas=1.2, R_gas=300, Pr_gas=0.7, rib_thickness=0.001):
        """
        Initialize the regenerative cooling channel sizer.
        :param contour_array: Nx2 array [x (m), r (m)] for chamber contour (axial position, radius)
        :param t_wall: Fixed inner wall thickness (m)
        :param fluid_coolant: Coolant fluid name for CoolProp (e.g., 'Oxygen')
        :param P_inlet: Coolant inlet pressure (MPa)
        :param T_inlet: Coolant inlet temperature (K)
        :param mdot_coolant: Total coolant mass flow rate (kg/s)
        :param T_gas: Combustion gas temperature (K) - assume constant for simplicity
        :param P_chamber: Chamber pressure (MPa)
        :param k_wall: Wall thermal conductivity (W/m-K)
        :param T_wall_max: Maximum allowable wall temperature (K)
        :param N_channels: Number of cooling channels (fixed)
        :param h_channel_initial: Initial guess for channel height (m)
        :param material: Wall material (for reference)
        :param gamma_gas: Gas specific heat ratio
        :param R_gas: Gas specific gas constant (J/kg-K)
        :param Pr_gas: Gas Prandtl number
        :param rib_thickness: Fixed rib (fin) thickness between channels (m)
        """
        self.contour = contour_array
        self.t_wall = t_wall
        self.fluid = fluid_coolant
        self.P_in = P_inlet
        self.T_in = T_inlet
        self.mdot = mdot_coolant
        self.Tg = T_gas
        self.Pc = P_chamber
        self.k_wall = k_wall
        self.Tw_max = T_wall_max
        self.N = N_channels
        self.h_ch_init = h_channel_initial
        self.material = material
        self.gamma = gamma_gas
        self.R = R_gas
        self.Pr = Pr_gas
        self.rib_thickness = rib_thickness

        # Discretize contour into axial stations
        self.dx = np.diff(self.contour[:, 0])
        self.x_stations = self.contour[:-1, 0] + self.dx / 2
        self.r_stations = (self.contour[:-1, 1] + self.contour[1:, 1]) / 2
        self.L_total = self.contour[-1, 0] - self.contour[0, 0]

        # Precompute throat
        self.r_throat = np.min(self.contour[:, 1])
        self.x_throat = self.contour[np.argmin(self.contour[:, 1]), 0]
        self.A_throat = np.pi * self.r_throat ** 2

        # Initialize arrays
        self.n_stations = len(self.x_stations)
        self.q = np.zeros(self.n_stations)  # Heat flux (W/m²)
        self.T_c = np.zeros(self.n_stations)  # Coolant temperature
        self.P_c = np.zeros(self.n_stations)  # Coolant pressure
        self.T_wg = np.zeros(self.n_stations)  # Gas-side wall temp
        self.T_wc = np.zeros(self.n_stations)  # Coolant-side wall temp
        self.h_g = np.zeros(self.n_stations)  # Gas-side HTC
        self.h_c = np.zeros(self.n_stations)  # Coolant-side HTC
        self.b_ch = np.zeros(self.n_stations)  # Channel width
        self.h_ch = np.zeros(self.n_stations)  # Channel height (will be constant)
        self.dP = np.zeros(self.n_stations)  # Pressure drop per station

    def calculate_mach(self, i):
        """
        Calculate Mach number at station i using isentropic relation.
        """
        A_local = np.pi * self.r_stations[i] ** 2
        area_ratio = A_local / self.A_throat

        def func(M):
            term = (2 / (self.gamma + 1)) * (1 + (self.gamma - 1) / 2 * M ** 2)
            power = (self.gamma + 1) / (2 * (self.gamma - 1))
            return (1 / M) * (term ** power) - area_ratio

        if self.x_stations[i] < self.x_throat:
            M_guess = 0.5
        else:
            M_guess = 2.0

        M = fsolve(func, np.array([M_guess]))[0]
        return M

    def gas_side_htc(self, i):
        """
        Gas-side heat transfer coefficient using Bartz equation.
        """
        D = 2 * self.r_stations[i]
        mu0 = 1e-5  # Reference viscosity, approximate for combustion gas
        Cp0 = self.gamma * self.R / (self.gamma - 1)  # Approximate Cp
        Pr0 = self.Pr
        T0 = self.Tg  # Stagnation temp approx as Tg
        rho_g = self.Pc * 1e6 / (self.R * self.Tg)  # Density, P in Pa

        M = self.calculate_mach(i)
        Tg_local = T0 / (1 + (self.gamma - 1) / 2 * M ** 2)  # Recovery temp
        ug = M * np.sqrt(self.gamma * self.R * Tg_local)

        # Bartz sigma factor
        sigma = (0.5 * (self.T_wg[i] / T0) * (1 + (self.gamma - 1) / 2 * M ** 2) + 0.5) ** (-0.68) * \
                (1 + (self.gamma - 1) / 2 * M ** 2) ** (-0.12)

        h_g = 0.026 / D ** 0.2 * (Cp0 * mu0 ** 0.2 / Pr0 ** 0.6) * (rho_g * ug) ** 0.8 * sigma
        return h_g

    def coolant_side_htc(self, i, h_ch):
        """
        Coolant-side HTC for rectangular channel using Dittus-Boelter.
        """
        perimeter = np.pi * 2 * self.r_stations[i]  # Inner perimeter
        self.b_ch[i] = perimeter / self.N - self.rib_thickness  # Channel width to keep rib constant
        a_ch = self.b_ch[i] * h_ch  # Area per channel
        mdot_ch = self.mdot / self.N
        rho_c = get_fluid_props(self.fluid, self.P_c[i], self.T_c[i], 'D')
        v_c = mdot_ch / (rho_c * a_ch)
        Dh = 2 * self.b_ch[i] * h_ch / (self.b_ch[i] + h_ch)

        mu_c = get_fluid_props(self.fluid, self.P_c[i], self.T_c[i], 'V')
        Cp_c = get_fluid_props(self.fluid, self.P_c[i], self.T_c[i], 'C')
        k_c = get_fluid_props(self.fluid, self.P_c[i], self.T_c[i], 'L')

        Pr_c = mu_c * Cp_c / k_c
        Re_c = rho_c * v_c * Dh / mu_c
        Nu = 0.023 * Re_c ** 0.8 * Pr_c ** 0.4  # Turbulent assumption
        h_c = Nu * k_c / Dh
        return h_c

    def pressure_drop(self, i, h_ch):
        """
        Friction pressure drop per station using Darcy-Weisbach.
        """
        a_ch = self.b_ch[i] * h_ch
        mdot_ch = self.mdot / self.N
        rho_c = get_fluid_props(self.fluid, self.P_c[i], self.T_c[i], 'D')
        v_c = mdot_ch / (rho_c * a_ch)
        Dh = 2 * self.b_ch[i] * h_ch / (self.b_ch[i] + h_ch)
        mu_c = get_fluid_props(self.fluid, self.P_c[i], self.T_c[i], 'V')
        Re = rho_c * v_c * Dh / mu_c
        f = 0.316 / Re ** 0.25 if Re > 3000 else 64 / Re
        dP = f * self.dx[i] / Dh * (rho_c * v_c ** 2 / 2)
        return dP

    def compute_profile(self, h_ch_const):
        """
        Compute the thermal and pressure profile for a given constant channel height.
        Returns the maximum gas-side wall temperature.
        """
        self.h_ch[:] = h_ch_const

        self.T_c[-1] = self.T_in
        self.P_c[-1] = self.P_in

        for i in range(self.n_stations - 1, -1, -1):
            def equations(vars):
                T_wg, T_wc, q = vars
                self.T_wg[i] = T_wg  # Temporary for h_g
                h_g = self.gas_side_htc(i)
                T_aw = self.Tg  # Approx
                eq1 = q - h_g * (T_aw - T_wg)
                eq2 = q - self.k_wall / self.t_wall * (T_wg - T_wc)
                h_c = self.coolant_side_htc(i, self.h_ch[i])
                eq3 = q - h_c * (T_wc - self.T_c[i])
                return np.array([eq1, eq2, eq3])

            guess = np.array([self.Tw_max, self.Tw_max - 50, 1e6])
            sol = fsolve(equations, guess, xtol=1e-6)
            self.T_wg[i], self.T_wc[i], self.q[i] = sol

            self.h_g[i] = self.gas_side_htc(i)
            self.h_c[i] = self.coolant_side_htc(i, self.h_ch[i])

            self.dP[i] = self.pressure_drop(i, self.h_ch[i])

            if i > 0:
                A_surf = np.pi * 2 * self.r_stations[i] * self.dx[i]
                dQ = self.q[i] * A_surf
                Cp_c = get_fluid_props(self.fluid, self.P_c[i], self.T_c[i], 'C')
                dT = dQ / (self.mdot * Cp_c)
                self.T_c[i - 1] = self.T_c[i] + dT  # T increases in flow direction (towards head)

                self.P_c[i - 1] = self.P_c[i] - self.dP[i]  # P decreases in flow direction

        return np.max(self.T_wg)

    def size_channels(self):
        """
        Size the constant channel height to ensure max wall temp equals T_wall_max.
        Channel width varies to keep rib thickness constant.
        """

        def objective(h_ch):
            max_t = self.compute_profile(h_ch[0])
            return np.array([max_t - self.Tw_max])

        h_ch_sol = fsolve(objective, np.array([self.h_ch_init]), xtol=1e-6)[0]

        # Run once more to set the final profile
        self.compute_profile(h_ch_sol)

        self.P_out_calc = self.P_c[0]  # Outlet at head
        self.T_out_calc = self.T_c[0]

    def plot_channel_cross_section(self, i):
        """
        Plot the cross-section of a single cooling channel at station i.
        Shows the inner wall (gas side), channel, and fin.
        """
        b_ch = self.b_ch[i]
        h_ch = self.h_ch[i]
        t_fin = self.rib_thickness  # Fixed rib thickness

        fig, ax = plt.subplots(figsize=(6, 4))

        # Gas side wall (inner liner)
        ax.add_patch(plt.Rectangle((0, 0), t_fin + b_ch + t_fin, self.t_wall, fill=None, edgecolor='black',
                                   label='Inner Wall (t_wall)'))

        # Channel (coolant passage)
        ax.add_patch(plt.Rectangle((t_fin, 0), b_ch, -h_ch, color='blue', alpha=0.5, label='Coolant Channel'))

        # Fins (ribs between channels)
        ax.add_patch(plt.Rectangle((0, 0), t_fin, -h_ch, color='gray', label='Fin'))
        ax.add_patch(plt.Rectangle((t_fin + b_ch, 0), t_fin, -h_ch, color='gray'))

        # Labels
        ax.text(t_fin + b_ch / 2, -h_ch / 2, f'Channel\nwidth: {b_ch * 1000:.2f} mm\nheight: {h_ch * 1000:.2f} mm',
                ha='center', va='center')
        ax.text((t_fin + b_ch + t_fin) / 2, self.t_wall / 2, f'Wall thickness: {self.t_wall * 1000:.2f} mm',
                ha='center', va='center')

        ax.set_xlim(0, t_fin + b_ch + t_fin)
        ax.set_ylim(-h_ch - 0.001, self.t_wall + 0.001)
        ax.set_aspect('equal')
        ax.set_xlabel('Circumferential Direction (m)')
        ax.set_ylabel('Radial Direction (m) [negative: outward]')
        ax.set_title(f'Channel Cross-Section at x = {self.x_stations[i]:.2f} m')
        ax.legend()
        plt.show()

    def plot_results(self):
        fig, axs = plt.subplots(4, 1, figsize=(10, 20))

        # Chamber contour plot
        axs[0].plot(self.contour[:, 0], self.contour[:, 1], label='Chamber Contour')
        axs[0].set_ylabel('Radius (m)')
        axs[0].legend()

        axs[1].plot(self.x_stations, self.T_wg, label='T wall gas')
        axs[1].plot(self.x_stations, self.T_wc, label='T wall coolant')
        axs[1].axhline(self.Tw_max, color='r', linestyle='--', label='Max T')
        axs[1].set_ylabel('Wall Temperature (K)')
        axs[1].legend()

        axs[2].plot(self.x_stations, self.h_ch * 1000, label='Channel Height (mm)')
        axs[2].set_ylabel('Channel Height (mm)')

        axs[3].plot(self.x_stations, self.q / 1e6, label='Heat Flux (MW/m²)')
        axs[3].set_ylabel('Heat Flux (MW/m²)')

        for ax in axs:
            ax.set_xlabel('Axial Position (m)')

        plt.tight_layout()
        plt.show()


# Example usage
# Define a simple bell nozzle contour (x, r in meters)
x = np.linspace(0, 1.0, 50)  # Axial from 0 to 1m
r_throat = 0.05
r_chamber = 0.1
r_exit = 0.2
# Simple approximation for contour
r = r_chamber * np.ones_like(x)
r[x >= 0.2] = r_throat + (r_exit - r_throat) * ((x[x >= 0.2] - 0.2) / 0.8) ** 1.5
r[x < 0.2] = r_chamber - (r_chamber - r_throat) * ((0.2 - x[x < 0.2]) / 0.2) ** 2
contour = np.column_stack((x, r))

sizer = RegenCoolingSizer(contour, t_wall=0.001, fluid_coolant='Oxygen', P_inlet=10, T_inlet=100,
                          mdot_coolant=5, T_gas=3000, P_chamber=5, k_wall=20, T_wall_max=800,
                          N_channels=100, h_channel_initial=0.005)

sizer.size_channels()
sizer.plot_results()

# Plot channel cross-section at throat (approximate index)
idx_throat = np.argmin(sizer.r_stations)
sizer.plot_channel_cross_section(idx_throat)

print("Calculated outlet pressure at head:", sizer.P_out_calc, "MPa")
print("Outlet coolant temperature at head:", sizer.T_out_calc, "K")