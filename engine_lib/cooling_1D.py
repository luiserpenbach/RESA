"""
NOTATION

Channel Parameters:
w_ch = Width of cooling channel
h_ch = Height of cooling channel
w_rib = Width of rib
t_iw = Inner wall thickness
t_ow = Outer wall thicness

i = Station (Engine gets divided in axial positions (stations) at which cooling properties are calculated)
A_i = Inner area of cooling channel station
delta_p_ch = Coolant Pressure drop for a station

Engine Paramers
p_c = Chamber chamber


@author: Luis
"""

import numpy as np
import CoolProp.CoolProp as CP
from rocketprops.rocket_prop import get_prop  # for n2o data
import matplotlib.pyplot as plt
import math
from rocketcea.cea_obj_w_units import CEA_Obj as CEA_Obj_units
from rocketcea.cea_obj import CEA_Obj


T_W_G = 400

class Regen_Cooling:
    def __init__(self, engine, stations, surface_roughness, t_iw=1, t_ow=3, wall_conductivity=330):

        print("Regen Cooling Class Initialized!")

        self.stations = stations  # Resolution for solving

        self.CEA = engine.CEA

        self.p_c = engine.p_c
        self.T_c = engine.T_c
        self.MR = engine.MR
        self.ER = engine.ER
        self.c_star = engine.c_star

        self.M_vec = engine.M_vec
        self.sM_vec = np.zeros(stations)  # M_vec reduced to discrete stations
        self.AR_vec = engine.AR_vec
        self.sAR_vec = np.zeros(stations)

        self.T_vec = engine.T_vec
        self.p_vec = engine.p_vec
        self.sT_vec = np.zeros(stations)  # Hot gas temp at stations
        self.sp_vec = np.zeros(stations)  # Pressure at stations

        self.massflow_f = engine.massflow_f
        self.massflow_ox = engine.massflow_ox

        # Wall data
        self.k = wall_conductivity  # Wall material thermal conductivity (maybe use array for temperature dependancy)
        self.t_iw = t_iw  # Inner wall thickness
        self.t_ow = t_ow  # Outer wall thickness
        self.tc_contourx = engine.tc_contourx
        self.tc_contoury = engine.tc_contoury
        self.rho_chamber = 8.9 * 1e3  # 8900 kg/m^3 for CuCrZr
        self.chamber_mass = 0
        self.surface_roughness = surface_roughness

        self.R_c = engine.R_c
        self.R_t = engine.R_t

        self.Pr_c = engine.Pr_c
        self.Cp_c = engine.Cp_c
        self.mu_c = engine.mu_c

        self.gamma = engine.gamma  # Ratio of specific heats

        # Analysis Results
        self.T_w_g_result = np.zeros(stations)
        self.T_w_c_result = np.zeros(stations)
        self.T_co_result = np.zeros(stations)
        self.p_co_result = np.zeros(stations)
        self.v_co_result = np.zeros(stations)
        self.q_result = np.zeros(stations)
        self.h_g_result = np.zeros(stations)
        self.w_ch_result = np.zeros(stations)  # mm!
        self.X_co_result = np.zeros(stations) # Fluid Quality


        self.Q_tot = 0
        self.num_channels = 0

        self.g = 9.81

    def get_number_channels(self, w_ch, w_rib):
        # calculates fitting number of channels based on channel and rib width
        R = self.R_c + self.t_iw  # Radial position of cooling channel base
        theta_channel = 2 * np.arcsin(w_ch / (2 * R))  # central angle from chamber center to channel cord length
        theta_rib = 2 * np.arcsin(w_rib / (2 * R))
        theta = theta_channel + theta_rib
        num_channels = math.floor(2 * np.pi / theta)  # how many thetas fit in full circle

        return num_channels

    def get_adapted_channel_width(self, r_chb, num_channels, w_rib):
        # calc local channel width based on rib width, channel number and local chamber radius
        # r_chb = local radius to channel base
        print(r_chb)
        # Angular span of rib (using arcsin chord formula)
        theta_rib = 2 * np.arcsin(w_rib / (2 * r_chb)) # [rad]

        # Angular span per sector (channel + rib)
        theta_sector = 2 * np.pi / num_channels # [rad]

        # Angular span of open channel
        theta_ch = theta_sector - theta_rib
        # Convert back to chord length
        w_ch = 2 * r_chb * np.sin(theta_ch / 2)

        return w_ch

    def reduce_contour_resolution_old(self, stations):  # works
        # divide contour data into stations
        # take start and end point
        # divide into sections -> stations-1
        # move steps while x < x_end
        og_len = len(self.tc_contourx)

        if stations > og_len:  # check if too many stations
            stations = og_len
            print("Too many stations! Value was reduced to ", stations)
        station_size = og_len // (stations - 1)  # right ??
        station_x = np.zeros(stations)
        station_y = np.zeros(stations)

        station_x[-1] = self.tc_contourx[-1]  # set last station
        station_y[-1] = self.tc_contoury[-1]
        self.sM_vec[-1] = self.M_vec[-1]
        self.sAR_vec[-1] = self.AR_vec[-1]
        self.sT_vec[-1] = self.T_vec[-1]
        self.sp_vec[-1] = self.p_vec[-1]

        # fill up stations in middle
        for k in range(0, stations - 1):
            station_x[k] = self.tc_contourx[k * station_size]
            station_y[k] = self.tc_contoury[k * station_size]

            # fill up new property arrays (Mach, ?)
            self.sM_vec[k] = self.M_vec[k * station_size]
            self.sAR_vec[k] = self.AR_vec[k * station_size]
            self.sT_vec[k] = self.T_vec[k * station_size]
            self.sp_vec[k] = self.p_vec[k * station_size]

        return station_x, station_y

    def reduce_contour_resolution(self, stations):
        og_len = len(self.tc_contourx)

        if stations > og_len:  # check if too many stations
            stations = og_len
            print("Too many stations! Value was reduced to ", stations)

        # Assume self.tc_contourx is sorted ascending (axial coordinate)
        # New station x: uniformly spaced in x (better than uniform index)
        station_x = np.linspace(self.tc_contourx[0], self.tc_contourx[-1], stations)

        # Interpolate y (radius) at new x positions
        station_y = np.interp(station_x, self.tc_contourx, self.tc_contoury)

        # Interpolate flow properties at new x positions
        # (creates new arrays — assumes originals are aligned with tc_contourx)
        self.sM_vec = np.interp(station_x, self.tc_contourx, self.M_vec)
        self.sAR_vec = np.interp(station_x, self.tc_contourx, self.AR_vec)
        self.sT_vec = np.interp(station_x, self.tc_contourx, self.T_vec)
        self.sp_vec = np.interp(station_x, self.tc_contourx, self.p_vec)

        return station_x, station_y

    def solve_channel(self, coolant, massflow, p_co_0, T_co_0, w_ch0=1, h_ch0=1.5, w_rib=1):
        ##### COMPARE WITH CIRA VALUES
        """
        Notation:
        co = coolant
        inf = Steady-state

        T_co = Temperature, Coolant
        T_w_g = Temperature, Wall Gas Side
        T_w_co = Temperature, Wall Coolant Side
        T_w = Temperature, Wall
        """

        # Update coolant if it is Ethanol-Water mixture
        if coolant == "Ethanol80":
            water_content = 20  # % of fuel
            coolant = 'Ethanol[{}]&Water[{}]'.format(1 - water_content / 100, water_content / 100)

        elif coolant == "Ethanol90":
            water_content = 10  # % of fuel
            coolant = 'Ethanol[{}]&Water[{}]'.format(1 - water_content / 100, water_content / 100)

        p_co_0 *= 1e5  # Bar -> Pa

        station_x, station_y = self.reduce_contour_resolution(self.stations)
        self.num_channels = self.get_number_channels(w_ch0, w_rib)
        print("CHANNEL NUMBER: ", self.num_channels)

        # Surface roughness for cooling channel
        # roughness = 6 * 1e-6  # m

        # Coolant massflow per channel
        mdot_co = massflow / self.num_channels

        # Set initial wall temperatures EFFECT ?????????
        T_w_g = T_W_G    # K (Gas side to steady-state estimation)
        T_w_c = T_co_0  # K (Coolant side to coolant temp)

        # self.w_ch_result[0] = w_ch0
        cross_area0 = w_ch0 * h_ch0 * 1e-6  # Channel inlet area, m^2

        # Set initial fluid properties
        p_co = p_co_0
        T_co = T_co_0

        rho_co = CP.PropsSI('D', 'T', T_co, 'P', p_co, coolant)
        cp_co = CP.PropsSI("CPMASS", 'T', T_co, 'P', p_co, coolant)

        v_co = 0  # m/s
        self.Q_tot = 0  # Total Heat
        self.chamber_mass = 0  # kg

        ch_length = 0.01

        # solve for temperatures at each station
        for x in range(0, self.stations):
            print("Station ", x, "of ", self.stations)

            # --- LOCAL GEOMETRY

            # Channel station length
            if (x < self.stations - 1):  # dont calc length of last station
                delta_x = (station_x[x + 1] - station_x[x]) * 1e-3
                delta_y = (station_y[x + 1] - station_y[x]) * 1e-3
                L = math.sqrt(delta_x ** 2 + delta_y ** 2)  # Station length, m

            # local chamber dimensions, m (channel height, rib width, wall thickness stay constant)
            y = station_y[x] * 1e-3  # m
            t_iw = self.t_iw * 1e-3  # m
            t_ow = self.t_ow * 1e-3  # m
            h_ch = h_ch0 * 1e-3  # m
            w_ch = self.get_adapted_channel_width(y + t_iw, self.num_channels,
                                                  w_rib * 1e-3)  # adapting channel width, m
            self.w_ch_result[x] = w_ch * 1e3  # save in mm
            # print("-- Local Channel width:", w_ch*1e3)

            D_h = self.hydraulic_dia_rect(w_ch, h_ch)  # m
            cross_area = w_ch * h_ch  # Channel cross section area, m^2

            # --- COOLANT FLOW PROPERTIES

            # Calculate dynamic pressure and temp from previous station
            p_dyn_1 = 0.5 * rho_co * v_co ** 2
            T_dyn_1 = 0.5 * v_co ** 2 / cp_co

            # Update coolant density [kg/m^3] and flow velocity [m/s]
            rho_co = CP.PropsSI('D', 'T', T_co, 'P', p_co, coolant)
            v_co = mdot_co / (rho_co * cross_area)  # (accelerate fluid due to new area and density)

            # Calculate dynamic pressure and temp of current station
            p_dyn_2 = 0.5 * rho_co * v_co ** 2
            T_dyn_2 = 0.5 * v_co ** 2 / cp_co

            p_co = p_co +0# (p_dyn_2 - p_dyn_1)
            T_co = T_co - (T_dyn_2 - T_dyn_1)
            # print("-- Dynamic Temperature Drop: ", T_dyn_2 - T_dyn_1)

            # Calculate coolant thermo properties
            cp_co = CP.PropsSI("CPMASS", 'T', T_co, 'P', p_co, coolant) # Cp mass specific [J/(kg*K)]
            mu_co = CP.PropsSI("V", 'T', T_co, 'P', p_co, coolant) # Dynamic Viscosity [Pa*s]
            k_co = CP.PropsSI("L", 'T', T_co, 'P', p_co, coolant) # Conductivity [W/(m*K)]
            X_co = CP.PropsSI("Q", 'T', T_co, 'P', p_co, coolant) # Quality [-]

            if T_co < 270:
                print("N2O temperature very low!")
            elif T_co > 600:
                print("N2O temperature very high!")

            # Reynolds Number
            Re_co = rho_co * v_co * D_h / mu_co
            # Prandtl Number
            Pr_co = mu_co * cp_co / k_co

            # print("-- Prandtl number: ", Pr_co)
            # print("-- Reynolds number: ", Re_co)

            # Friction factor, f0 for when wall would be smooth
            f = self.get_f(self.surface_roughness, D_h, Re_co)
            f0 = self.get_f(0, D_h, Re_co)
            # Surface roughness effect correction
            ksi = f / f0
            psi = (1 + 1.5 * Pr_co ** (-1 / 6) * Re_co ** (-1 / 8) * (Pr_co - 1)) / (
                    1 + 1.5 * Pr_co ** (-1 / 6) * Re_co ** (-1 / 8) * (Pr_co * ksi - 1)) * ksi

            # --- HOT GAS FLOW PROPERTIES

            T_g = self.sT_vec[x]  # local hot gas temp
            T_aw = self.get_T_aw(T_g, Pr_co, self.gamma, self.sM_vec[x])
            Pr_g = self.Pr_c  # interpolate ??
            Cp_g = self.Cp_c  # ''
            mu_g = self.mu_c  # ''

            T_w_g_new = T_w_g + 10  # avoid missing the loop
            T_w_c_new = T_w_c + 10

            # --- ITERATE UNTIL TEMP CONVERGES
            while (abs(T_w_g_new - T_w_g) > 0.1) and (abs(T_w_c_new - T_w_c) > 0.1):
                T_w_g = T_w_g_new
                T_w_c = T_w_c_new

                # Calc convective heat coeff
                h_g = self.bartz(self.T_c, T_w_g, self.p_c, self.sM_vec[x], self.R_t * 2 * 1e-3, self.sAR_vec[x], mu_g,
                                 Cp_g, Pr_g, self.gamma, self.c_star)

                # Calc Nusselt Number (maybe use Taylor approach later) -> Coolant heat transfer coefficient
                Nu_co = self.dittus_boelter(Re_co, Pr_co)
                Nu_co_taylor = self.Taylor(Re_co, Pr_co, T_co, T_w_c, D_h, ch_length)
                # print("-- Nusselt D-B: ", Nu_co)
                # print("-- Nusselt Taylor: ", Nu_co_taylor)
                h_co = (k_co * Nu_co / D_h) * psi
                # Rib/Fin effectiveness correction
                eta_f = np.tanh(np.sqrt(2.0 * h_co * (w_rib * 1e-3) / self.k) * h_ch / (w_rib * 1e-3)) / np.sqrt(
                    2 * h_co * (w_rib * 1e-3) / self.k) * h_ch / (w_rib * 1e-3)  # w_rib in mm!
                h_co = h_co * (w_ch + 2.0 * eta_f * h_ch) / (w_ch + (w_rib * 1e-3))

                # Radiative Heat transfer approximation
                p_h2o = 1  # partial pressure of h2o in combustion gas
                p_co2 = 1  # partial pressure of co2
                q_r_h2o = 5.74 * (p_h2o / 1e5 * y) ** 0.3 * (T_g / 100) ** 3.5
                q_r_co2 = 4 * (p_co2 / 1e5 * y ** 0.3 * (
                        T_g / 100) ** 3.5)  # Radiation Heat Flux (get CO2, H2O pressures from CEA)
                q_rad = q_r_h2o + q_r_co2
                q_rad = 0  # ADD IN LATER

                # Calc heat flux
                q = (T_aw - T_co + q_rad / h_g) / (1 / h_g + t_iw / self.k + 1 / h_co)  # t_iw in m!

                # Wall temperatures
                T_w_g_new = T_aw - (q - q_rad) / h_g
                T_w_c_new = T_co + q / h_co

            T_w_g = T_w_g_new
            T_w_c = T_w_c_new

            # --- UPDATE COOLANT PRESSURE AND TEMPERATURE

            # Total Pressure drop ( Darcy-Weisbach and Dynamic pressure change)
            delta_p_co = f * rho_co * L / D_h * v_co ** 2 / 2.0
            p_co = p_co - delta_p_co
            print("--  delta p = ", delta_p_co * 1e-5 * 1e3, " mBar")

            A_i = 2.0 * np.pi * y * L  # Inner Chamber area of station that gets heated, m^2
            A_c_i = A_i / self.num_channels  # Effective area per channel, m^2

            # Temperature increase
            delta_T_co = q * A_c_i / (mdot_co * cp_co)
            T_co += delta_T_co

            # Update total heat
            self.Q_tot += q * A_i

            # Update Channel Length
            ch_length += L

            # Chamber weight estimation
            m_station = (2.0 * np.pi * y * L * (t_iw + t_ow) + L * (
                    w_rib * 1e-3) * h_ch * self.num_channels) * self.rho_chamber
            self.chamber_mass += m_station

            # Insert into result arrays
            self.T_co_result[x] = T_co
            self.p_co_result[x] = p_co
            self.T_w_g_result[x] = T_w_g
            self.T_w_c_result[x] = T_w_c
            self.q_result[x] = q
            self.h_g_result[x] = h_g
            self.v_co_result[x] = v_co
            self.X_co_result[x] = X_co
            print("QUALITY: ", X_co)
            # print("--  T_w_g = ", T_w_g, " K")
            # print("--  q = ", q*1e-3, " kW/m^2")

        print("Total heat: ", self.Q_tot * 1e-3)
        print("Estimated chamber mass: ", self.chamber_mass, " kg")
        print("Heat transfer analysis finished!")

    def hydraulic_dia_rect(self, w, h):  # Hydraulic diameter for rectangular pipe
        print("WIDTH, ", w)
        return 2.0 * w * h / (w + h)

    def dittus_boelter(self, Re, Pr):  # Dittus Boelter Correlation for Nusselt Number
        return 0.023 * Re ** 0.8 * Pr ** 0.4

    def get_T_aw(self, T_g, Pr, gamma, M):
        # Adiabtic wall temperature
        r = Pr ** (1 / 3)  # Recovery factor for turbulent flow
        T_aw = T_g * ((1.0 + r*0.5*(gamma - 1.0)*M**2) / (1+0.5 * (gamma-1) * M**2))

        return T_aw

    def get_f(self, roughness, D, Re):
        """ Gouder-Sonnad approximation to directly solve Darcy-Weisbach equation (friction factor f, dimensionless)
        """
        ep = roughness
        a = 2.0 / np.log(10)
        b = ep / D / 3.7
        d = np.log(10) * Re / 5.02
        s = b * d + np.log(d)
        q = s * (s / (s + 1))
        g = b * d + np.log(d / q)
        z = np.log(q / g)
        D_la = z * g / (g + 1.0)
        D_cfa = D_la * (1.0 + z / 2.0 * ((g + 1.0) ** 2.0 + (z / 3.0) * (2.0 * g - 1.0)))

        f = np.sqrt(1.0 / (a * (np.log(d / g) + D_cfa)))
        return f

    def bartz(self, T_c, T_w, p_c, M, D_t, AR, mu, cp, Pr, gamma, c_star):
        """
        Calculate convective heat transfer coefficients along engine axis using Bartz equation
        T_c = Total Combustion Temperature
        """

        # Transport parameters
        Pr_c = Pr
        Cp_c = cp
        mu_c = mu

        # Conversion into imperial units
        c_star_I = c_star * 3.23  # [ft/s]
        Cp_c_I = 0.238846 * Cp_c  # [BTU/(lb*F°)]
        D_t_I = 39.3701 * D_t  # D_t in m! [in]
        g_I = 3.28 * self.g  # [ft/s^2]
        p_c_I = 0.0001450377 * p_c * 1.01325e5  # [lb/in^2]
        mu_c_I = 5.6e-6 * mu_c  # [lb/(in*s)]

        # Boundary layer correction factor at each X-point
        sigma = 1 / ((0.5 * T_w / T_c * (1 + ((gamma - 1) / 2) * M ** 2) + 0.5) ** 0.68 * (
                1 + ((gamma - 1) / 2) * M ** 2) ** 0.12)

        # Heat transfer coefficient vector using Bartz equation [BTU(s*in^2*F)]
        h_g_I = ((0.026 / D_t_I ** 0.2) * (((mu_c_I ** 0.2) * Cp_c_I) / Pr_c ** 0.6) * (
                (p_c_I * g_I) / c_star_I) ** 0.8) * ((1 / AR) ** 0.9) * sigma
        # Conversion to metric units [W/(m^2*K)]
        h_g = 20441.748028012 * 144 * h_g_I * 0.5  # 0.65 correction factor to compensate overprediction of Bartz (From Delft MA)

        return h_g

    def Taylor(self, Re, Pr, T_co, T_w, Dh, ch_length):
        h_g = 0.023 * Re ** 0.8 * Pr ** 0.4 * (T_co / T_w) ** (0.57 - 1.59 * Dh / ch_length)
        return h_g

    def plot_bartz(self):
        x, y = self.reduce_contour_resolution(self.stations)  # get x, y coordinates of stations
        fig, ax = plt.subplots()
        plt.grid(True, color="dimgray", linestyle='dotted')
        # plt.fill()
        # ax.set_aspect('equal')
        ax.set_xlabel("X Position [mm]")
        ax.set_ylabel("Heat Transfer Coeff. [W/(m^2*K)]")
        ax.set_title("Bartz Approximation - h_g")
        ax.plot(x, self.h_g_result, label='Bartz h_g', color="green")
        plt.show()

    def multiplot(self):
        x, y = self.reduce_contour_resolution(self.stations)  # get x, y coordinates of stations

        fig, axs = plt.subplots(3, 3)
        # plt.fill()
        # ax.set_aspect('equal')

        plt.grid(True, color="dimgray", linestyle='dotted')

        plt.text(4, 0, "Number Channels: " + str(self.num_channels), ha="left", wrap=True)

        # Contour, Bartz, Mach
        axs[0, 0].set_aspect("equal")
        axs[0, 0].set_ylabel("Y")
        axs[0, 0].set_title(f"Chamber Contour Discretization ({self.stations} Stations)")
        axs[0, 0].plot(x, y, "bo", label='Contour', color="white")
        axs[0, 0].plot(x, -y, label='Contour Negative', color="White")
        # for i in range(self.stations):
        #    axs[0, 0].axvline(x = x[i], color = 'grey', label = 'Station')

        axs[0, 1].set_ylabel("h_g [kW/(m^2*K)]")
        axs[0, 1].set_title("Heat transfer coefficient (gas-side, Bartz)")
        axs[0, 1].plot(x, self.h_g_result * 1e-3, "-", label='h_g', color="red")

        axs[0, 2].set_ylabel("q [MW/m^2]")
        axs[0, 2].set_title("Heat Flux")
        axs[0, 2].plot(x, self.q_result * 1e-6, label='q', color="blue", )
        import pandas as pd
        dictt = {"x [mm]": x, "q [MW/m^2]": self.q_result * 1e-6}
        df = pd.DataFrame(dictt)
        # saving the dataframe
        df.to_csv('heat_data.csv')

        # Wall Temps, Coolant Temp
        axs[1, 0].set_ylabel("T_w_g [K]")
        axs[1, 0].set_title("Wall Temperature, gas-side")
        axs[1, 0].plot(x, self.T_w_g_result, label='T_w_g', color="orange")

        axs[1, 1].set_ylabel("T_w_c [K]")
        axs[1, 1].set_title("Wall Temp, coolant-side")
        axs[1, 1].plot(x, self.T_w_c_result, label='T_w_c', color="orange")

        axs[1, 2].set_ylabel("T_co [K]")
        axs[1, 2].set_title("Coolant Temp")
        axs[1, 2].plot(x, self.T_co_result, label='T_co', color="hotpink")

        # Coolant pressure, Velocity, Heat Flux
        axs[2, 0].set_xlabel("Axial Position")
        axs[2, 0].set_ylabel("p_co [bar]")
        axs[2, 0].set_title("Coolant Pressure")
        axs[2, 0].plot(x, self.p_co_result * 1e-5, label='p_co', color="blue")

        axs[2, 1].set_xlabel("Axial Position")
        axs[2, 1].set_ylabel("v_co [m/s]")
        axs[2, 1].set_title("Flow velocity")
        axs[2, 1].plot(x, self.v_co_result, label='v_co', color="blue")

        #ALTERNATIVE
        #axs[2, 2].set_xlabel("")
        #axs[2, 2].set_ylabel("w_ch [mm]")
        #axs[2, 2].set_title("Local Channel Width")
        #axs[2, 2].plot(x, self.w_ch_result, label='w_ch', color="olive")
        axs[2, 2].set_xlabel("Axial Position [mm]")
        axs[2, 2].set_ylabel("X [-]")
        axs[2, 2].set_title("Quality")
        axs[2, 2].plot(x, self.X_co_result, label='w_ch', color="olive")

        # turn on grid on all subplots
        for ax in axs.flat:
            ax.grid(True, color="dimgray", linestyle='dotted')

        manager = plt.get_current_fig_manager()
        #manager.full_screen_toggle()
        plt.show()
