import numpy as np
from CoolProp.CoolProp import AbstractState, PropsSI


'''
Swirl Injector Sizing Functions
----------------------------------------
This is code originally written by Moritz vom Schemm during his Master Thesis. 
It has been edited to be more modular and improve readability by Constantin Bleul. 
See Moritz vom Schemm's Master Thesis for further details.
'''

def get_uppercase_x(alpha : float):
    '''
    Gets dimensionless gas centre area X
    Valid for alpha between 40 and 60 degrees
    :param alpha: Spraywinkel in degrees
    :return: Dimensionless gas centre area
    '''
    if alpha < 40 or alpha > 60:
        raise ValueError('alpha must be between 40° and 60°')
    return 0.0042 * alpha ** 1.2714


def get_c_d_maximum_flow(X):
    '''
    Gets discharge coefficient according to Nardi et al. assuming maximum flow.
    See equation 5.10 in MA vom Schemm.
    :param X: Dimensionless gas centre area
    :return: Discharge Coefficient
    '''

    return np.sqrt(((1 - X) ** 3) / (1 + X))


def calculate_density(fluid: str = None,
                      pressure:float = None,
                      temperature:float = None,
                      state: AbstractState = None) -> float:
    """
    Calculates density using CoolProp.
    Handles two input options: CoolProp Abstract State or pressure, temperature and fluid inputs

    :param fluid:           -           CoolProp fluid name e.g. "N2O"
    :param pressure:        Pa          Pressure
    :param temperature:     K           Temperature
    :param state:           -           CoolProp Abstract State
    :return:                kg/m^3      Density
    """


    rho: float

    if state is not None:
        try:
            rho = state.rhomass()
        except Exception as e:
            raise ValueError(f'Error getting density from Abstract State: {e}')

        return rho

    elif pressure is not None and temperature is not None and fluid is not None:
        try:
            rho = PropsSI('D', 'P', pressure, 'T', temperature, fluid)
        except Exception as e:
            raise ValueError(f'PropsSI call failed: {e}')
        return rho
    else:
        raise ValueError('Either fluid AbstractState or pressure, temperature and fluid name are needed.')

def get_heat_capacity_ratio(fluid: str = None,
                            pressure:float = None,
                            temperature:float = None,
                            state: AbstractState = None) -> float:

    """
    Returns heat capacity ratio using CoolProp.
    Handles two input options: CoolProp Abstract State or pressure, temperature and fluid inputs

    :param fluid:                       CoolProp fluid name e.g. "N2O"
    :param pressure:        Pa          Pressure
    :param temperature:     K           Temperature
    :param state:                       CoolProp Abstract State
    :return:                kg/m^3      Density
    """

    if state is not None:
        try:
            cp = state.cpmass()
            cv = state.cvmass()
        except Exception as e:
            raise ValueError(f'Error getting density from Abstract State: {e}')

        return cp/cv

    elif pressure is not None and temperature is not None and fluid is not None:
        try:
            cp = PropsSI('Cpmass', 'P', pressure, 'T', temperature, fluid)
            cv = PropsSI('Cvmass', 'P', pressure, 'T', temperature, fluid)
        except Exception as e:
            raise ValueError(f'PropsSI call failed: {e}')
        return cp / cv
    else:
        raise ValueError('Either fluid AbstractState or pressure, temperature and fluid name are needed.')

def get_specific_gas_constant(fluid: str = None):

    try:
        state = AbstractState("HEOS", fluid)
        universal_gas_constant = PropsSI(fluid, 'gas_constant')
    except Exception as e:
        raise ValueError(f'Error calling CoolProp. Check fluid name: {e}')

    return universal_gas_constant / state.molar_mass()




#-------------------------------------------------------------
# Not yet sorted or edited

def get_c_d_abramovic(r_sc, r_p, n_p, r_o = None):

    if r_o == None:
        r_o = r_sc
    A = (r_sc - r_p) * r_o / (n_p * r_p ** 2)   # Swirl number
    C_D_Abramovich = 0.432 * A ** -0.64         # discharge coefficient derived by Abramovic

    return C_D_Abramovich


def get_c_d_rizk(r_sc, r_p, n_p, r_o = None):
    if r_o == None:
        r_o = r_sc
    A_tot_p = n_p * np.pi * r_p ** 2                                                # total tangential port area
    C_D_Rizk = 0.35 * np.sqrt(A_tot_p / (4 * r_sc * r_o)) * (r_sc / r_o) ** 0.25    # discharge coefficient derived by Rizk and Lefebvre 1985

    return C_D_Rizk


def get_c_d_hong(r_sc, r_p, n_p, r_o = None):
    if r_o == None:
        r_o = r_sc
    beta = (r_sc - r_p) / r_o
    A_tot_p = n_p * np.pi * r_p ** 2                                                        # total tangential port area
    C_D_Hong = 0.44 * (A_tot_p / (4 * r_o ** 2)) ** (0.84 * beta ** -0.52) * beta ** -0.59  # discharge coefficient derived by Hong et al

    return C_D_Hong


def get_c_d_fu(r_sc, r_p, n_p):
    A = (r_sc - r_p) * r_sc / (n_p * r_p ** 2)  # Swirl number
    C_D_Fu = 0.4354 * A ** -0.877               # discharge coefficient derived by Fu et al

    return C_D_Fu


def get_c_d_anand(r_sc, r_p, n_p):
    A = (r_sc - r_p) * r_sc / (n_p * r_p ** 2)  # Swirl number
    C_D_Anand = 1.28 * A ** -1.28               # discharge coefficient derived by Anand et al

    return C_D_Anand


def get_alpha_lefebvre(X):
    alpha_lefebvre = np.arcsin(X * np.sqrt(8) / ((1 + np.sqrt(X)) * np.sqrt(1 + X)))

    return alpha_lefebvre


def get_alpha_anand(r_sc, r_p, n_p, RR):
    A = (r_sc - r_p) * r_sc / (n_p * r_p ** 2)                  # Swirl number
    alpha_Anand = np.arctan(0.01 * A ** 1.64 * RR ** -0.242)    # spray angle derived by Anand et al

    return alpha_Anand


def get_alpha_fu(m_dot_f, eta_f, r_sc, r_p, n_p, r_o = None):
    if r_o == None:
        r_o = r_sc
    Re_p = 2 * m_dot_f / (np.pi * r_p * eta_f * np.sqrt(n_p))   # Reynolds number port
    A = (r_sc - r_p) * r_o / (n_p * r_p ** 2)                   # Swirl number
    alpha_Fu = np.arctan(0.033 * A ** 0.338 * Re_p ** 0.249)

    return alpha_Fu


def get_film_thickness_fu(m_dot_f, eta_f, rho_f, del_p, r_sc):
    t_film_fu = 3.1 * ((2 * r_sc * m_dot_f * eta_f) / (rho_f * del_p)) ** 0.25       # film thickness for open end swirl injectors derived by Fu et al based on correlation of Rizk and Lefebvre

    return t_film_fu


def get_film_thickness_suyari_lefebvre(m_dot_f, eta_f, rho_f, del_p, r_o):
    t_film_suyari_lefebvre = 2.7 * ((2 * r_o * m_dot_f * eta_f) / (rho_f * del_p)) ** 0.25   # film thickness derived by Suyari and Lefebvre based on correlation of Rizk and Lefebvre

    return t_film_suyari_lefebvre


def get_film_thickness_simmons_harding(m_dot_f, alpha, rho_f, del_p, r_sc):
    t_film_simmons_harding = (0.00805 * np.sqrt(rho_f) * m_dot_f) / (np.sqrt(del_p * rho_f) * 2 * r_sc * np.cos(np.deg2rad(alpha)))

    return t_film_simmons_harding