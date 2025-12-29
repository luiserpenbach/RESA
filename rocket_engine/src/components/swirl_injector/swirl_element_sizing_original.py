"""
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


def swirl_calculator_rezende(alpha, del_p, T_f, T_ox, m_dot_f, m_dot_ox, v_ox, n, n_p, t_post, p, oxidizer, fuel):
    # Calculate propellant properties from given pressure and temperature
    propellant_props = list(get_propellant_conditions(T_f, T_ox, p, oxidizer, fuel))
    propellant_props_cc = list(get_propellant_conditions(T_f, T_ox, p - del_p, oxidizer, fuel))
    propellant_props_thermo = list(get_propellant_conditions_thermo(T_f, T_ox, p - del_p))

    rho_f_inj = propellant_props[0]
    rho_ox_inj = propellant_props[1]
    eta_f_inj = propellant_props[2]
    M_ox = propellant_props[3]
    rho_f_cc = propellant_props_cc[0]
    rho_ox_cc = propellant_props_cc[1]
    eta_f_cc = propellant_props_cc[2]
    gamma_ox = propellant_props_cc[4]
    sigma_f_cc = propellant_props_thermo[3]

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
    A_ox_orifice = m_dot_ox / (p / np.sqrt(T_ox) * np.sqrt(gamma_ox * M_ox / 8.314) *
                               ((gamma_ox + 1) / 2) ** (
                                           (-gamma_ox + 1) / (2 * (gamma_ox - 1))))  # oxidizer orifice area
    A_ox = m_dot_ox / (v_ox * rho_ox_cc * n)  # oxidizer port area
    r_ox = np.sqrt(A_ox / np.pi + (r_o + t_post) ** 2)  # oxidizer port radius
    r_ox_orifice = np.sqrt(A_ox_orifice / np.pi)  # oxidizer orifice radius

    # Calculate injector properties
    SN = (r_sc - r_p) * r_sc / (n_p * r_p ** 2)
    r_aircore = X * r_o
    v_f_ax = m_dot_f / (n * rho_f_cc * np.pi * (r_o ** 2 - r_aircore ** 2))  # axial fuel velocity
    J = (rho_ox_cc * v_ox ** 2) / (rho_f_cc * v_f_ax ** 2)  # momentum flux ratio
    RV = v_ox / v_f_ax  # velocity ratio
    Re_p = 2 * m_dot_f / (np.pi * r_p * eta_f_inj * np.sqrt(n_p))  # Reynolds number port
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
    propellant_props = list(get_propellant_conditions(T_f, T_ox, p, oxidizer, fuel))
    propellant_props_cc = list(get_propellant_conditions(T_f, T_ox, p - del_p, oxidizer, fuel))
    propellant_props_thermo = list(get_propellant_conditions_thermo(T_f, T_ox, p - del_p))

    rho_f_inj = propellant_props[0]
    rho_ox_inj = propellant_props[1]
    eta_f_inj = propellant_props[2]
    M_ox = propellant_props[3]
    rho_f_cc = propellant_props_cc[0]
    rho_ox_cc = propellant_props_cc[1]
    eta_f_cc = propellant_props_cc[2]
    gamma_ox = propellant_props_cc[4]
    sigma_f_cc = propellant_props_thermo[3]

    # Sizing of central oxidizer port
    A_ox_orifice = m_dot_ox / (p / np.sqrt(T_ox) * np.sqrt(gamma_ox * M_ox / 8.314) *
                               ((gamma_ox + 1) / 2) ** (
                                           (-gamma_ox + 1) / (2 * (gamma_ox - 1))))  # oxidizer orifice area
    A_ox = m_dot_ox / (v_ox * rho_ox_cc * n)  # oxidizer port area
    r_ox = np.sqrt(A_ox / np.pi)  # oxidizer port radius
    r_ox_orifice = np.sqrt(A_ox_orifice / np.pi)  # oxidizer orifice radius

    r_p = 0.0004  # start value for port radius
    minimum_clearence = 0.0005  # minimum clearance required by printer
    m_dot_f_current = 0  # mass flow for current geometry

    # Size geometry to fit required mass flow
    while abs(m_dot_f_current - m_dot_f) / m_dot_f > 0.01:
        r_o = r_ox + t_post + minimum_clearence  # injector orifice radius
        t_film = get_film_thickness_fu(m_dot_f, eta_f_cc, rho_f_cc, del_p, r_o)  # fuel film thickness

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
    Re_p = 2 * m_dot_f / (np.pi * r_p * eta_f_inj * np.sqrt(n_p))  # Reynolds number port
    beta = (r_o - r_p) / r_o  # swirl chamber - orifice radius ratio
    FN = m_dot_ox / (n * np.sqrt(del_p * rho_ox_inj))  # Flow number (m^2)
    alpha = np.rad2deg(alpha)  # convert spray half angle to degrees
    We = (rho_ox_cc * ((v_ox - v_f_ax) ** 2) * t_film) / sigma_f_cc  # Weber number

    print('Spray half angle = ' + str(round(alpha, 2)) + ' °')
    print('Swirl number = ' + str(round(SN, 2)))
    print('Momentum flux ratio = ' + str(round(J, 2)))
    print('Velocity ratio = ' + str(round(RV, 2)))
    print('Weber number = ' + str(round(We, 2)))
    print('Inlet port radius = ' + str(round(r_p * 1000, 2)) + ' mm')
    print('Injector orifice radius = ' + str(round(r_o * 1000, 2)) + ' mm')
    print('Film thickness = ' + str(round(t_film * 1000, 2)) + ' mm')
    print('Oxidizer port radius = ' + str(round(r_ox * 1000, 2)) + ' mm')
    print('Oxidizer port orifice radius = ' + str(round(r_ox_orifice * 1000, 2)) + ' mm')
    print('Element fuel mass flow = ' + str(round(m_dot_element, 4)) + ' kg/s')
    print('Element oxidizer mass flow = ' + str(round(m_dot_ox / n, 4)) + ' kg/s')
    print('Oxidizer density = ' + str(round(rho_ox_cc, 2)) + ' kg/m^3')
    print('Fuel density = ' + str(round(rho_f_cc, 2)) + ' kg/m^3')
    print('Fuel surface tension = ' + str(round(sigma_f_cc, 4)) + ' N/m')
    print()
    print('Spray half angle after Fu = ' + str(
        round(np.rad2deg(get_alpha_fu(m_dot_f, eta_f_inj, r_o, r_p, n_p)), 2)) + ' °')
    print('Discharge Coefficient after Hong = ' + str(round(get_c_d_hong(r_o, r_p, n_p), 4)))
    print('Discharge Coefficient after Fu = ' + str(round(get_c_d_fu(r_o, r_p, n_p), 4)))
    print('Discharge Coefficient after Anand = ' + str(round(get_c_d_anand(r_o, r_p, n_p), 4)))

    return J, r_p, r_o, r_ox, C_D


def swirl_calculator_cold_flow(J, r_p, r_sc, r_o, r_ox, n_p, p_in, del_p, t_post, C_D, gcsc=True):
    # Calculate fluid properties from given pressure and temperature
    T_amb = 293.15
    fluid_props = list(get_propellant_conditions(T_amb, T_amb, p_in, 'nitrogen', 'water'))
    fluid_props_amb = list(get_propellant_conditions(T_amb, T_amb, p_in - del_p, 'nitrogen', 'water'))
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
        alpha = get_alpha_anand(r_sc, r_p, n_p, 1.5)
    else:
        m_dot_n2 = rho_n2_amb * v_n2 * np.pi * (r_n2 - t_post - r_o) ** 2
        X = r_aircore ** 2 / r_o ** 2
        alpha = get_alpha_lefebvre(X)

    A_n2_orifice = m_dot_n2 / (p_in / np.sqrt(T_amb) * np.sqrt(gamma_n2 * M_n2 / 8.314) *
                               ((gamma_n2 + 1) / 2) ** (
                                           (-gamma_n2 + 1) / (2 * (gamma_n2 - 1))))  # oxidizer orifice area
    r_n2_orifice = np.sqrt(A_n2_orifice / np.pi)  # oxidizer orifice radius

    SN = (r_sc - r_p) * r_sc / (n_p * r_p ** 2)  # Swirl number
    We = (rho_n2_amb * ((v_n2 - v_w_ax) ** 2) * t_film) / sigma_w_amb  # Weber number

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

    return r_sc, r_o, r_p, r_n2, r_n2_orifice


'''
def injector_performance(r_sc, r_o, r_p, r_ox, r_ox_orifice, t_post, n, n_p, del_p_f, del_p_ox, p_cc, T_ox, T_f, oxidizer, fuel, gcsc = True):
    # Calculate propellant properties
    propellant_props_ox_inj = list(get_propellant_conditions(T_f, T_ox, p_cc + del_p_ox, oxidizer, fuel))
    propellant_props_f_inj = list(get_propellant_conditions(T_f, T_ox, p_cc + del_p_f, oxidizer, fuel))
    propellant_props_cc = list(get_propellant_conditions(T_f, T_ox, p_cc, oxidizer, fuel))
    propellant_props_thermo = list(get_propellant_conditions_thermo(T_f, T_ox, p_cc))
    rho_f_inj = propellant_props_f_inj[0]
    rho_ox_inj = propellant_props_ox_inj[1]
    eta_f_inj = propellant_props_f_inj[2]
    rho_f_cc = propellant_props_cc[0]
    rho_ox_cc = propellant_props_cc[1]
    eta_f_cc = propellant_props_cc[2]
    M_ox = propellant_props_cc[3]
    gamma_ox = propellant_props_cc[4]
    sigma_f_cc = propellant_props_thermo[3]

    A_ox_orifice = np.pi * r_ox_orifice ** 2
    A_o = np.pi * r_o ** 2
    m_dot_ox = A_ox_orifice * (del_p_ox + p_cc) / np.sqrt(T_ox) * np.sqrt(gamma_ox * M_ox / 8.314) *\
               ((gamma_ox + 1) / 2) ** ((gamma_ox + 1) / (2 * (gamma_ox - 1)))

    if gcsc:
        A_ox = np.pi * r_ox ** 2
        C_D_f = get_c_d_anand(r_sc, r_p, n_p)
        m_dot_f = C_D_f * A_o * np.sqrt(2 * rho_f_inj * del_p_f)
        alpha = get_alpha_anand(r_o, r_p, n_p, 1.5)                                 # Spray half angle
        t_film = get_film_thickness_fu(m_dot_f, eta_f_cc, rho_f_cc, del_p_f, r_sc)

    else:
        A_ox = np.pi * (r_ox ** 2 - (r_o + t_post) ** 2)
        C_D_f = get_c_d_rizk(r_sc, r_p, n_p, r_o)
        m_dot_f = C_D_f * A_o * np.sqrt(2 * rho_f_inj * del_p_f)
        t_film = get_film_thickness_suyari_lefebvre(m_dot_f, eta_f_cc, rho_f_cc, del_p_f, r_sc)
        X = (r_o - t_film) ** 2 / r_o ** 2
        alpha = get_alpha_lefebvre(X)

    # Calculate injector properties
    v_ox = m_dot_ox / (rho_ox_cc * A_ox)                                        # oxidizer velocity
    r_aircore = r_o - t_film                                                    # central aircore radius
    ROF = m_dot_ox / m_dot_f                                                    # ratio oxidizer / fuel
    v_f_ax = m_dot_f / (n * rho_f_cc * np.pi * (r_o ** 2 - r_aircore ** 2))     # axial fuel velocity
    J = (rho_ox_cc * v_ox ** 2) / (rho_f_cc * v_f_ax ** 2)                      # momentum flux ratio
    Re_p = 2 * m_dot_f / (np.pi * r_p * eta_f_inj * np.sqrt(n_p))               # Reynolds number port
    alpha = np.rad2deg(alpha)                                                   # convert spray half angle to degrees
    We = (rho_ox_cc * (v_ox ** 2) * t_film) / sigma_f_cc                        # Weber number

    return ROF, J, We, alpha
'''


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
    alpha_lefebvre = np.arcsin(X * np.sqrt(8) / (1 + np.sqrt(X) * np.sqrt(1 + X)))

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


def get_propellant_conditions(T_f, T_ox, p, oxidizer, fuel):
    rho_f = CP.PropsSI('D', 'T', T_f, 'P', p, fuel)
    rho_ox = CP.PropsSI('D', 'T', T_ox, 'P', p, oxidizer)
    eta_f = CP.PropsSI('V', 'T', T_f, 'P', p, fuel)
    M_ox = CP.PropsSI('M', oxidizer)
    gamma_ox = CP.PropsSI('isentropic_expansion_coefficient', 'T', T_ox, 'P', p, oxidizer)

    return rho_f, rho_ox, eta_f, M_ox, gamma_ox


def get_propellant_conditions_thermo(T_f, T_ox, p):
    oxidizer = Chemical('Nitrous Oxide')
    oxidizer.calculate(T_ox, p)
    fuel = Mixture(['ethanol', 'water'], ws=[0.8, 0.2], T=T_f, P=p)

    # PFUSCH
    water = Chemical('water')
    ethanol = Chemical('ethanol')
    water.calculate(T_f, p)
    ethanol.calculate(T_f, p)

    rho_f = fuel.rho
    rho_ox = oxidizer.rho
    eta_f = fuel.mu
    sigma_f = fuel.sigma

    # PFUSCH
    sigma_f = water.sigma * 0.2 + ethanol.sigma * 0.8

    return rho_f, rho_ox, eta_f, sigma_f


if __name__ == "__main__":
    # propellant properties

    Temperature_fuel = 300  # fuel inlet temperature (K)
    Temperature_oxidizer = 500  # oxidizer inlet temperature (K)
    Fuel = 'REFPROP::Ethanol'
    Oxidizer = 'REFPROP::NitrousOxide'
    Pressure_inlet = 45e5  # injector inlet pressure (Pa)
    Pressure_drop = 20e5  # injector pressure drop (Pa)

    Mass_flow_fuel = 0.20  # ethanol mass flow (kg/s)
    Velocity_oxidizer_lcsc = 100  # oxidizer exit velocity from injector (m/s)
    Mass_flow_oxidizer = 0.8  # nitrous oxide mass flow (kg/s)
    Elements = 3  # number of injector elements
    Number_of_ports = 3  # number of tangential ports per injector element
    Post_thickness = 0.5*1e-3  # thickness of oxidizer post (m)
    Spray_half_angle = 60  # spray half angle produced by injector - design parameter (°)

    # Calculations
    '''
    print('Calculation of gas centered swirl coaxial injector with correlations by Anand et al.:')
    Output_real_props_gcsc = swirl_calculator_anand(Post_thickness, Velocity_oxidizer_gcsc, Mass_flow_fuel,
                                                    Mass_flow_oxidizer, Elements, Number_of_ports,
                                                    Temperature_fuel, Temperature_oxidizer, Pressure_inlet,
                                                    Pressure_drop, Oxidizer, Fuel)

    print()
    print('Equivalent cold flow swirl injector:')
    Output_cold_flow_gcsc = swirl_calculator_cold_flow(Output_real_props_gcsc[0], Output_real_props_gcsc[1],
                                                       Output_real_props_gcsc[2], Output_real_props_gcsc[2],
                                                       Output_real_props_gcsc[3], Number_of_ports, Pressure_drop + 1e5,
                                                       Pressure_drop, Post_thickness, Output_real_props_gcsc[4])

    print()'''
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

    # a = injector_performance(Output_cold_flow_lcsc[0], Output_cold_flow_lcsc[1], Output_cold_flow_lcsc[2], Output_cold_flow_lcsc[3], Output_cold_flow_lcsc[4], Post_thickness, Elements, Number_of_ports, Pressure_drop, Pressure_drop, 1e5, 293.15, 293.15, 'nitrogen', 'water', gcsc=False)
    # b = injector_performance(Output_cold_flow_gcsc[0], Output_cold_flow_gcsc[1], Output_cold_flow_gcsc[2], Output_cold_flow_gcsc[3], Output_cold_flow_gcsc[4], Post_thickness, Elements, Number_of_ports, Pressure_drop, Pressure_drop, 1e5, 293.15, 293.15, 'nitrogen', 'water', gcsc=True)
