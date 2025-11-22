# -*- coding: utf-8 -*-
"""
                File name: Swirl_calc.py
                Author: Moritz vom Schemm
                Date created: 21.07.2023
                Date last modified: 10.08.2023
                Python Version: 3.7
                LCSC injector dimensioning after: Nardi, Rene; Neto, Sergio; Pimenta, Amilcar; Perez, Vladia (2014):
                Dimensioning a Simplex Swirl Injector. In: 50th AIAA/ASME/SAE/ASEE Joint Propulsion Conference.
                50th AIAA/ASME/SAE/ASEE Joint Propulsion Conference. Cleveland, OH. Reston,
                Virginia: American Institute of Aeronautics and Astronautics.
                modified to include a coaxial gasous oxidizer flow

                GCSC injector dimsensioning using correlations by Anand
"""
import numpy as np
import CoolProp.CoolProp as CP
from thermo import *
from scipy.optimize import fsolve


class GCSC_Injector:

    def __init__(self, t_post, v_ox, m_dot_f, m_dot_ox, n, n_p, T_f, T_ox, p, del_p, oxidizer, fuel):
        # Fuel mass flow
        self.m_dot_f = m_dot_f
        # Oxidizer mass flow
        self.m_dot_ox = m_dot_ox
        # Number of injector elements
        self.n = n
        # Number of ports
        self.n_p = n_p
        # Oxidizer
        self.oxidizer = oxidizer
        # Fuel
        self.fuel = fuel
        # Injector entry pressure
        self.p = p
        # Injector pressure drop
        self.del_p = del_p
        # Fuel entry temperature
        self.T_f = T_f
        # Oxidizer entry temperature
        self.T_ox = T_ox
        # post thickness
        self.t_post = t_post
        # oxidizer exit velocity
        self.v_ox = v_ox


def swirl_calculator_rezende(alpha, del_p, T_f, T_ox, m_dot_f, m_dot_ox, v_ox, n, n_p, t_post, p, oxidizer, fuel):
    # Calculate propellant properties from given pressure and temperature
    propellant_props = list(get_fluid_conditions(T_f, T_ox, p, oxidizer, fuel))
    propellant_props_cc = list(get_fluid_conditions(T_f, T_ox, p - del_p, oxidizer, fuel))

    rho_f_inj = propellant_props[0]
    rho_ox_inj = propellant_props[1]
    eta_f_inj = propellant_props[2]
    M_ox = propellant_props[3]
    rho_f_cc = propellant_props_cc[0]
    rho_ox_cc = propellant_props_cc[1]
    eta_f_cc = propellant_props_cc[2]
    gamma_ox = propellant_props_cc[4]
    surface_tension_eth = CP.PropsSI('I', 'T', T_f, 'Q', 0, 'Ethanol')
    surface_tension_w = CP.PropsSI('I', 'T', T_f, 'Q', 0, 'Water')
    sigma_f_cc = (surface_tension_w * 0.2 + surface_tension_eth * 0.8)

    # Calculate open area ratio A_aircore / A_exit from spray angle
    X = 0.0042 * alpha ** 1.2714

    # Calculate orifice size
    C_D = get_c_d_maximum_flow(X)  # injector discharge coefficient
    A_tot_e = m_dot_f / (C_D * np.sqrt(2 * rho_f_inj * del_p))  # total injector area
    r_o = np.sqrt(4 * A_tot_e / (np.pi * n)) / 2  # swirl orifice radius

    # Calculate port size
    C_D_p = np.sqrt(X ** 3 / (2 - X))  # tangential port discharge coefficient
    A_tot_p = m_dot_f / (n * C_D_p * np.sqrt(2 * rho_f_inj * del_p))  # total tangential port area
    r_p = np.sqrt(A_tot_p / (np.pi * n_p))  # tangential port radius

    # Calculate swirl chamber geometry
    r_sc = 3.3 * r_o  # radius swirl chamber
    l_sc = 2 * r_sc  # length swirl chamber
    l_o = r_o  # orifice length

    # Calculate oxidizer orifice
    C_D_orifice = 0.65  # sharp-edged orifice
    A_ox_orifice = m_dot_ox / (n * C_D_orifice * np.sqrt(2 * rho_ox_inj * del_p))  # oxidizer orifice area
    A_ox = m_dot_ox / (v_ox * rho_ox_cc * n)  # oxidizer port area
    r_ox = np.sqrt(A_ox / np.pi + (r_o + t_post) ** 2)  # oxidizer port radius
    r_ox_orifice = np.sqrt(A_ox_orifice / np.pi)  # oxidizer orifice radius

    # Calculate injector properties
    SN = (r_sc - r_p) * r_sc / (n_p * r_p ** 2)
    r_aircore = np.sqrt(X) * r_o
    v_f_ax = m_dot_f / (n * rho_f_cc * np.pi * (r_o ** 2 - r_aircore ** 2))  # axial fuel velocity
    J = (rho_ox_cc * v_ox ** 2) / (rho_f_cc * v_f_ax ** 2)  # momentum flux ratio
    RV = v_ox / v_f_ax  # velocity ratio
    Re_p = 2 * m_dot_f / (n * np.pi * r_p * eta_f_inj * np.sqrt(n_p))  # Reynolds number port
    beta = (r_sc - r_p) / r_o
    FN = m_dot_ox / (n * np.sqrt(del_p * rho_ox_inj))  # Flow number (m^2)
    t_film = r_o * (1 - np.sqrt(X))  # Film thickness (m)
    We = (rho_ox_cc * ((v_ox - v_f_ax) ** 2) * t_film) / sigma_f_cc  # Weber number

    # Output
    print('Spray half angle = ' + str(round(alpha, 2)) + ' °')
    print('Swirl number = ' + str(round(SN, 2)))
    print('Momentum flux ratio = ' + str(round(J, 2)))
    print('Velocity ratio = ' + str(round(RV, 2)))
    print('Weber number = ' + str(round(We, 2)))
    print('Inlet port radius = ' + str(round(r_p * 1000, 2)) + ' mm')
    print('Injector orifice radius = ' + str(round(r_o * 1000, 2)) + ' mm')
    print('Oxidizer port radius = ' + str(round(r_ox * 1000, 2)) + ' mm')
    print('Oxidizer port orifice radius = ' + str(round(r_ox_orifice * 1000, 2)) + ' mm')
    print('Film thickness = ' + str(round(t_film * 1000, 2)) + ' mm')
    print('Axial fuel velocity = ' + str(round(v_f_ax, 2)) + ' m/s')
    print('Element fuel mass flow = ' + str(round(m_dot_f / n, 4)) + ' kg/s')
    print('Element oxidizer mass flow = ' + str(round(m_dot_ox / n, 4)) + ' kg/s')
    print('Oxidizer density = ' + str(round(rho_ox_cc, 2)) + ' kg/m^3')
    print('Fuel density = ' + str(round(rho_f_cc, 2)) + ' kg/m^3')
    print('Fuel surface tension = ' + str(round(sigma_f_cc, 4)) + ' N/m')
    print()
    print('Discharge Coefficient with maximum flow assumption = ' + str(round(get_c_d_maximum_flow(X), 4)))
    print('Discharge Coefficient after Abramovic = ' + str(round(get_c_d_abramovic(r_sc, r_p, n_p, r_o), 4)))
    print('Discharge Coefficient after Rizk = ' + str(round(get_c_d_rizk(r_sc, r_p, n_p, r_o), 4)))

    return J, r_p, r_sc, r_o, r_ox, C_D


def swirl_calculator_anand(t_post, v_ox, m_dot_f, m_dot_ox, n, n_p, T_f, T_ox, p, del_p, oxidizer, fuel):
    # Calculate propellant properties from given pressure and temperature
    propellant_props = list(get_fluid_conditions(T_f, T_ox, p, oxidizer, fuel))
    propellant_props_cc = list(get_fluid_conditions(T_f, T_ox, p - del_p, oxidizer, fuel))

    rho_f_inj = propellant_props[0]
    rho_ox_inj = propellant_props[1]
    eta_f_inj = propellant_props[2]
    M_ox = propellant_props[3]
    rho_f_cc = propellant_props_cc[0]
    rho_ox_cc = propellant_props_cc[1]
    eta_f_cc = propellant_props_cc[2]
    gamma_ox = propellant_props_cc[4]
    surface_tension_eth = CP.PropsSI('I', 'T', T_f, 'Q', 0, 'Ethanol')
    surface_tension_w = CP.PropsSI('I', 'T', T_f, 'Q', 0, 'Water')
    sigma_f_cc = (surface_tension_w * 0.2 + surface_tension_eth * 0.8)

    # Sizing of central oxidizer port
    C_D_orifice = 0.65  # sharp-edged orifice
    A_ox_orifice = m_dot_ox / (n * C_D_orifice * np.sqrt(2 * rho_ox_inj * del_p))  # oxidizer orifice area
    A_ox = m_dot_ox / (v_ox * rho_ox_cc * n)  # oxidizer port area
    r_ox = np.sqrt(A_ox / np.pi)  # oxidizer port radius
    r_ox_orifice = np.sqrt(A_ox_orifice / np.pi)  # oxidizer orifice radius

    r_p = 0.0004  # start value for port radius
    minimum_clearence = 0.0005  # minimum clearance required by printer
    m_dot_f_current = 0  # mass flow for current geometry

    # Size geometry to fit required mass flow
    while abs(m_dot_f_current - m_dot_f) / m_dot_f > 0.01:
        r_o = r_ox + t_post + minimum_clearence  # injector orifice radius
        t_film = get_film_thickness_fu(m_dot_f / n, eta_f_cc, rho_f_cc, del_p, r_o)  # fuel film thickness

        if t_film > minimum_clearence:
            t_film = minimum_clearence

        A_orifice = np.pi * r_o ** 2  # injector orifice area
        C_D = get_c_d_anand(r_o, r_p, n_p)  # Empircal correlation Anand et al
        m_dot_element = C_D * A_orifice * np.sqrt(2 * rho_f_inj * del_p)  # mass flow per element
        m_dot_f_current = n * m_dot_element  # total massflow
        r_p = r_p * np.sqrt(m_dot_f / m_dot_f_current)  # correction of port radius to achieve desired mass flow

    # Calculate injector properties
    SN = (r_o - r_p) * r_o / (n_p * r_p ** 2)  # Swirl number
    r_aircore = r_o - t_film  # central air core radius
    alpha = get_alpha_anand(r_o, r_p, n_p, 1.5)  # Spray half angle, Empircal correlation Anand et al
    v_f_ax = m_dot_f / (n * rho_f_cc * np.pi * (r_o ** 2 - r_aircore ** 2))  # axial fuel velocity
    J = (rho_ox_cc * v_ox ** 2) / (rho_f_cc * v_f_ax ** 2)  # momentum flux ratio
    RV = v_ox / v_f_ax  # velocity ratio
    Re_p = 2 * m_dot_f / (n * np.pi * r_p * eta_f_inj * np.sqrt(n_p))  # Reynolds number port
    beta = (r_o - r_p) / r_o  # swirl chamber - orifice radius ratio
    FN = m_dot_ox / (n * np.sqrt(del_p * rho_ox_inj))  # Flow number (m^2)
    alpha = np.rad2deg(alpha)  # convert spray half angle to degrees
    We = (rho_ox_cc * ((v_ox - v_f_ax) ** 2) * t_film) / sigma_f_cc  # Weber number

    print('Spray half angle = ' + str(round(np.rad2deg(get_alpha_fu(m_dot_f / n, eta_f_inj, r_o, r_p, n_p)), 2)) + ' °')
    print('Swirl number = ' + str(round(SN, 2)))
    print('Momentum flux ratio = ' + str(round(J, 2)))
    print('Velocity ratio = ' + str(round(RV, 2)))
    print('Weber number = ' + str(round(We, 2)))
    print('Inlet port radius = ' + str(round(r_p * 1000, 2)) + ' mm')
    print('Injector orifice radius = ' + str(round(r_o * 1000, 2)) + ' mm')
    print('Film thickness = ' + str(round(t_film * 1000, 2)) + ' mm')
    print('Axial fuel velocity = ' + str(round(v_f_ax, 2)) + ' m/s')
    print('Oxidizer port radius = ' + str(round(r_ox * 1000, 2)) + ' mm')
    print('Oxidizer port orifice radius = ' + str(round(r_ox_orifice * 1000, 2)) + ' mm')
    print('Element fuel mass flow = ' + str(round(m_dot_element, 4)) + ' kg/s')
    print('Element oxidizer mass flow = ' + str(round(m_dot_ox / n, 4)) + ' kg/s')
    print('Oxidizer density = ' + str(round(rho_ox_cc, 2)) + ' kg/m^3')
    print('Fuel density = ' + str(round(rho_f_cc, 2)) + ' kg/m^3')
    print('Fuel surface tension = ' + str(round(sigma_f_cc, 4)) + ' N/m')
    print()
    print('Spray half angle after Anand = ' + str(round(alpha, 2)) + ' °')
    print('Discharge Coefficient after Hong = ' + str(round(get_c_d_hong(r_o, r_p, n_p), 4)))
    print('Discharge Coefficient after Fu = ' + str(round(get_c_d_fu(r_o, r_p, n_p), 4)))
    print('Discharge Coefficient after Anand = ' + str(round(get_c_d_anand(r_o, r_p, n_p), 4)))

    '''
    m_dot_matrix = np.zeros((2,4))
    m_dot_matrix[0] = np.linspace(m_dot_f/(n * 3), m_dot_f/n, 4)
    m_dot_matrix[1] = np.linspace(m_dot_ox/(n * 3), m_dot_ox/n, 4)

    for i in range(len(m_dot_matrix[0])):
        print(np.divide(offdesign(C_D, A_ox_orifice, A_orifice, p - del_p, fuel, oxidizer, m_dot_matrix[0][i], m_dot_matrix[1][i], T_f, T_ox), 1e5))
    '''

    return J, r_p, r_o, r_ox, C_D


def swirl_calculator_cold_flow(J, r_p, r_sc, r_o, r_ox, n_p, p_in, del_p, t_post, C_D, gcsc=True):
    # Calculate fluid properties from given pressure and temperature
    T_amb = 293.15
    fluid_props = list(get_fluid_conditions(T_amb, T_amb, p_in, 'nitrogen', 'water'))
    fluid_props_amb = list(get_fluid_conditions(T_amb, T_amb, p_in - del_p, 'nitrogen', 'water'))
    rho_w_inj = fluid_props[0]
    rho_n2_inj = fluid_props[1]
    eta_w_inj = fluid_props[2]
    rho_w_amb = fluid_props_amb[0]
    rho_n2_amb = fluid_props_amb[1]
    eta_w_amb = fluid_props_amb[2]
    gamma_n2 = fluid_props_amb[4]
    M_n2 = fluid_props[3]
    water = Chemical('water')
    water.calculate(293.15, p_in - del_p)
    sigma_w_amb = water.sigma

    if gcsc:
        A_orifice = np.pi * r_o ** 2  # injector orifice area (m^2)
        m_dot_w = C_D * A_orifice * np.sqrt(2 * rho_w_inj * del_p)  # mass flow (kg/s)
        t_film = get_film_thickness_fu(m_dot_w, eta_w_amb, rho_w_amb, del_p, r_sc)  # film thickness (m)
        r_aircore = r_o - t_film  # air core radius (m)
        v_w_ax = m_dot_w / (rho_w_amb * np.pi * (r_o ** 2 - r_aircore ** 2))  # axial water velocity (m/s)
    else:
        A_orifice = np.pi * r_o ** 2  # injector orifice area
        m_dot_w = C_D * A_orifice * np.sqrt(2 * rho_w_inj * del_p)  # water mass flow
        t_film = get_film_thickness_suyari_lefebvre(m_dot_w, eta_w_amb, rho_w_amb, del_p, r_o)
        r_aircore = r_o - t_film
        v_w_ax = m_dot_w / (rho_w_amb * np.pi * (r_o ** 2 - r_aircore ** 2))

    # Sizing of nitrogen port depending on injector config
    v_n2 = np.sqrt(J * v_w_ax ** 2 * rho_w_amb / rho_n2_amb)
    r_n2 = r_ox
    if gcsc:
        m_dot_n2 = rho_n2_amb * v_n2 * np.pi * r_n2 ** 2
        alpha = get_alpha_fu(m_dot_w, eta_w_inj, r_sc, r_p, n_p, r_o)
    else:
        m_dot_n2 = rho_n2_amb * v_n2 * np.pi * (r_n2 - t_post - r_o) ** 2
        X = r_aircore ** 2 / r_o ** 2
        alpha = get_alpha_lefebvre(X)

    # check if flow is supersonic in orifice
    if p_in / (p_in - del_p) > 1.894:
        A_n2_orifice = m_dot_n2 / (p_in / np.sqrt(T_amb) * np.sqrt(gamma_n2 * M_n2 / 8.314) *
                                   ((gamma_n2 + 1) / 2) ** (
                                               -(gamma_n2 + 1) / (2 * (gamma_n2 - 1))))  # oxidizer orifice area
    else:
        C_D_orifice = 0.65  # sharp-edged orifice
        A_n2_orifice = m_dot_n2 / (C_D_orifice * np.sqrt(2 * rho_n2_inj * del_p))
    r_n2_orifice = np.sqrt(A_n2_orifice / np.pi)  # oxidizer orifice radius

    SN = (r_sc - r_p) * r_sc / (n_p * r_p ** 2)  # Swirl number
    We = (rho_n2_amb * ((v_n2 - v_w_ax) ** 2) * t_film) / sigma_w_amb  # Weber number

    m_dot_matrix = np.zeros((2, 4))
    p_matrix = np.zeros_like(m_dot_matrix)
    m_dot_matrix[0] = np.linspace(m_dot_w / 3, m_dot_w, 4)
    m_dot_matrix[1] = np.linspace(m_dot_n2 / 3, m_dot_n2, 4)
    A_gas = np.pi * 1.36e-3 ** 2

    for i in range(len(m_dot_matrix[0])):
        p_matrix[:, i] = offdesign(C_D, A_gas, A_orifice, p_in - del_p, 'water', 'nitrogen', m_dot_matrix[0][i],
                                   m_dot_matrix[1][i], T_amb, T_amb)

    print('Spray half angle = ' + str(round(np.rad2deg(alpha), 2)) + ' °')
    print('Swirl number = ' + str(round(SN, 2)))
    print('Momentum flux ratio = ' + str(round(J, 2)))
    print('Velocity ratio = ' + str(round(v_n2 / v_w_ax, 2)))
    print('Weber number = ' + str(round(We, 2)))
    print('Inlet port radius = ' + str(round(r_p * 1000, 2)) + ' mm')
    print('Swirl Chamber radius = ' + str(round(r_sc * 1000, 2)) + ' mm')
    print('Injector orifice radius = ' + str(round(r_o * 1000, 2)) + ' mm')
    print('Film thickness = ' + str(round(t_film * 1000, 2)) + ' mm')
    print('Nitrogen port radius = ' + str(round(r_n2 * 1000, 2)) + ' mm')
    print('Nitrogen orifice radius = ' + str(round(r_n2_orifice * 1000, 2)) + ' mm')
    print('Nitrogen velocity = ' + str(round(v_n2, 2)) + ' m/s')
    print('Element water mass flow = ' + str(round(m_dot_w, 4)) + ' kg/s')
    print('Element nitrogen mass flow = ' + str(round(m_dot_n2, 4)) + ' kg/s')
    print('Nitrogen density = ' + str(round(rho_n2_amb, 2)) + ' kg/m^3')
    print('Water density = ' + str(round(rho_w_amb, 2)) + ' kg/m^3')
    print('Water surface tension = ' + str(round(sigma_w_amb, 4)) + ' N/m')
    print('Operating points:')
    for i in range(len(m_dot_matrix)):
        if i == 0:
            print('Water:')
        else:
            print('Nitrogen:')
        for j in range(len(m_dot_matrix[0])):
            print(str(round(m_dot_matrix[i, j] * 1e3, 1)) + ' g/s,  \t' + str(round(p_matrix[i, j] / 1e5, 1)) + ' bar')

    return r_sc, r_o, r_p, r_n2, r_n2_orifice


def offdesign(C_D, A_gas, A_liquid, p, liquid, gas, m_dot_liquid, m_dot_gas, T_liquid, T_gas):
    fluid_props = list(get_fluid_conditions(T_liquid, T_gas, p, gas, liquid))
    rho_liquid = fluid_props[0]
    M_gas = fluid_props[3]
    gamma_gas = fluid_props[4]

    del_p_gas = m_dot_gas / (A_gas / np.sqrt(T_gas) * np.sqrt(gamma_gas * M_gas / 8.314) *
                             ((gamma_gas + 1) / 2) ** (-(gamma_gas + 1) / (2 * (gamma_gas - 1)))) - p
    del_p_liquid = 1 / (2 * rho_liquid) * (m_dot_liquid / (C_D * A_liquid)) ** 2

    p_inj_gas = del_p_gas + p
    # check if flow is subsonic in orifice
    if p_inj_gas / (p_inj_gas - del_p_gas) < 1.894:
        C_D_gas_orifice = 0.65
        temp = 0

        def func(p_inj):
            rho_gas = CP.PropsSI('D', 'T', T_gas, 'P', p_inj, gas)
            return 1 / (2 * rho_gas) * (m_dot_gas / (C_D_gas_orifice * A_gas)) ** 2 - (p_inj - p)

        # get del_p_gas
        p_inj_gas = fsolve(func, p_inj_gas)
        del_p_gas = p_inj_gas[0] - p

    return del_p_liquid, del_p_gas


def get_c_d_maximum_flow(X):
    C_D_maximum_flow = np.sqrt(((1 - X) ** 3) / (1 + X))  # discharge coefficient assuming maximum flow

    return C_D_maximum_flow


def get_c_d_abramovic(r_sc, r_p, n_p, r_o=None):
    if r_o == None:
        r_o = r_sc
    A = (r_sc - r_p) * r_o / (n_p * r_p ** 2)  # Swirl number
    C_D_Abramovich = 0.432 * A ** -0.64  # discharge coefficient derived by Abramovic

    return C_D_Abramovich


def get_c_d_rizk(r_sc, r_p, n_p, r_o=None):
    if r_o == None:
        r_o = r_sc
    A_tot_p = n_p * np.pi * r_p ** 2  # total tangential port area
    C_D_Rizk = 0.35 * np.sqrt(A_tot_p / (4 * r_sc * r_o)) * (
                r_sc / r_o) ** 0.25  # discharge coefficient derived by Rizk and Lefebvre 1985

    return C_D_Rizk


def get_c_d_hong(r_sc, r_p, n_p, r_o=None):
    if r_o == None:
        r_o = r_sc
    beta = (r_sc - r_p) / r_o
    A_tot_p = n_p * np.pi * r_p ** 2  # total tangential port area
    C_D_Hong = 0.44 * (A_tot_p / (4 * r_o ** 2)) ** (
                0.84 * beta ** -0.52) * beta ** -0.59  # discharge coefficient derived by Hong et al

    return C_D_Hong


def get_c_d_fu(r_sc, r_p, n_p):
    A = (r_sc - r_p) * r_sc / (n_p * r_p ** 2)  # Swirl number
    C_D_Fu = 0.4354 * A ** -0.877  # discharge coefficient derived by Fu et al

    return C_D_Fu


def get_c_d_anand(r_sc, r_p, n_p):
    A = (r_sc - r_p) * r_sc / (n_p * r_p ** 2)  # Swirl number
    C_D_Anand = 1.28 * A ** -1.28  # discharge coefficient derived by Anand et al

    return C_D_Anand


def get_alpha_lefebvre(X):
    alpha_lefebvre = np.arcsin(X * np.sqrt(8) / ((1 + np.sqrt(X)) * np.sqrt(1 + X)))

    return alpha_lefebvre


def get_alpha_anand(r_sc, r_p, n_p, RR):
    A = (r_sc - r_p) * r_sc / (n_p * r_p ** 2)  # Swirl number
    alpha_Anand = np.arctan(0.01 * A ** 1.64 * RR ** -0.242)  # spray angle derived by Anand et al

    return alpha_Anand


def get_alpha_fu(m_dot_f, eta_f, r_sc, r_p, n_p, r_o=None):
    if r_o == None:
        r_o = r_sc
    Re_p = 2 * m_dot_f / (np.pi * r_p * eta_f * np.sqrt(n_p))  # Reynolds number port
    A = (r_sc - r_p) * r_o / (n_p * r_p ** 2)  # Swirl number
    alpha_Fu = np.arctan(0.033 * A ** 0.338 * Re_p ** 0.249)

    return alpha_Fu


def get_film_thickness_fu(m_dot_f, eta_f, rho_f, del_p, r_sc):
    t_film_fu = 3.1 * ((2 * r_sc * m_dot_f * eta_f) / (
                rho_f * del_p)) ** 0.25  # film thickness for open end swirl injectors derived by Fu et al based on correlation of Rizk and Lefebvre

    return t_film_fu


def get_film_thickness_suyari_lefebvre(m_dot_f, eta_f, rho_f, del_p, r_o):
    t_film_suyari_lefebvre = 2.7 * ((2 * r_o * m_dot_f * eta_f) / (
                rho_f * del_p)) ** 0.25  # film thickness derived by Suyari and Lefebvre based on correlation of Rizk and Lefebvre

    return t_film_suyari_lefebvre


def get_film_thickness_simmons_harding(m_dot_f, alpha, rho_f, del_p, r_sc):
    t_film_simmons_harding = (0.00805 * np.sqrt(rho_f) * m_dot_f) / (
                np.sqrt(del_p * rho_f) * 2 * r_sc * np.cos(np.deg2rad(alpha)))

    return t_film_simmons_harding


def get_fluid_conditions(T_liquid, T_gas, p, gas, liquid):
    rho_liquid = CP.PropsSI('D', 'T', T_liquid, 'P', p, liquid)
    rho_gas = CP.PropsSI('D', 'T', T_gas, 'P', p, gas)
    eta_liquid = CP.PropsSI('V', 'T', T_liquid, 'P', p, liquid)
    M_gas = CP.PropsSI('M', gas)
    gamma_gas = CP.PropsSI('isentropic_expansion_coefficient', 'T', T_gas, 'P', p, gas)

    return rho_liquid, rho_gas, eta_liquid, M_gas, gamma_gas


if __name__ == "__main__":
    # propellant properties

    Temperature_fuel = 400  # fuel inlet temperature (K) 500
    Temperature_oxidizer = 600  # oxidizer inlet temperature (K) 600
    Fuel = 'Ethanol[0.8]&Water[0.2]'
    Oxidizer = 'NitrousOxide'
    Pressure_inlet = 75e5  # injector inlet pressure (Pa)
    Pressure_drop = 25e5  # injector pressure drop (Pa)

    Mass_flow_fuel = 0.25  # ethanol mass flow (kg/s)
    Velocity_oxidizer_gcsc = 100  # oxidizer exit velocity from injector (m/s)
    Velocity_oxidizer_lcsc = 100  # oxidizer exit velocity from injector (m/s)
    Mass_flow_oxidizer = 0.75  # nitrous oxide mass flow (kg/s)
    Elements = 3  # number of injector
    Number_of_ports = 3  # number of tangential ports
    Post_thickness = 0.0005  # thickness of oxidizer post (m)
    Spray_half_angle = 60  # spray half angle produced by injector - design parameter (°)

    # Calculations

    # print('Calculation of gas centered swirl coaxial injector with correlations by Anand et al.:')
    # Output_real_props_gcsc = swirl_calculator_anand(Post_thickness, Velocity_oxidizer_gcsc, Mass_flow_fuel, Mass_flow_oxidizer, Elements, Number_of_ports,
    # Temperature_fuel, Temperature_oxidizer, Pressure_inlet, Pressure_drop, Oxidizer, Fuel)

    # print()
    # print('Equivalent cold flow swirl injector:')
    # Output_cold_flow_gcsc = swirl_calculator_cold_flow(Output_real_props_gcsc[0], Output_real_props_gcsc[1], Output_real_props_gcsc[2], Output_real_props_gcsc[2], Output_real_props_gcsc[3], Number_of_ports, Pressure_drop + 1e5, Pressure_drop, Post_thickness, Output_real_props_gcsc[4])

    print()
    print('Calculation of liquid centered swirl coaxial injector with correlations by Rezende et al.:')
    Output_real_props_lcsc = swirl_calculator_rezende(Spray_half_angle, Pressure_drop, Temperature_fuel,
                                                      Temperature_oxidizer, Mass_flow_fuel, Mass_flow_oxidizer,
                                                      Velocity_oxidizer_lcsc, Elements, Number_of_ports, Post_thickness,
                                                      Pressure_inlet, Oxidizer, Fuel)

    print()
    print('Equivalent cold flow swirl injector:')
    Output_cold_flow_lcsc = swirl_calculator_cold_flow(Output_real_props_lcsc[0], Output_real_props_lcsc[1],
                                                       Output_real_props_lcsc[2], Output_real_props_lcsc[3],
                                                       Output_real_props_lcsc[4], Number_of_ports, Pressure_drop + 1e5,
                                                       Pressure_drop, Post_thickness, Output_real_props_lcsc[5],
                                                       gcsc=False)
