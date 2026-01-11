from typing import Union, List

from CoolProp.CoolProp import QT_INPUTS, PT_INPUTS, AbstractState, SmassT_INPUTS, PSmass_INPUTS, PropsSI
from numpy import pi, sqrt, array, ndarray, linspace, tan, deg2rad
from dataclasses import dataclass, field


from SwirlInjectorSizing import get_uppercase_x, get_c_d_maximum_flow, calculate_density, get_heat_capacity_ratio, \
    get_specific_gas_constant


@dataclass
class LCSCInjector:
    '''
    Klasse repräsentiert einen liquid centred swirl coaxial injector und enthält die geometrischen Größen.
    Genaueren Bedeutung der Variablen siehe MA Moritz vom Schemm Abbildung 5.4 oder in der vorgesehenen Notion-Seite
    '''

    # Größen definiert in MA vom Schemm

    d_ein: float                                    # Durchmesser tangential Einlass in die Swirl Kammer
    n_ein: int                                      # Anzahl tangentialer Einlässe in die Swirl Kammer
    d_aus: float                                    # Durchmesser des Swirl Kammer Auslasses
    l_aus: float                                    # Länge des zylindrischen Auslasses aus der Swirl Kammer
    d_ox: float                                     # Äußerer Durchmesser des Oxidator Ringspalts
    d_sk: float                                     # Durchmesser Swirl Kammer
    l_sk: float                                     # Länge des zylindrischen Teils der Swirl Kammer
    phi: float                                      # Aufgespannter Winkel aus Brennstoff und Oxidator Auslässen.
    post_wall_thickness: float                      # Wandstärke des Fuel Posts

    # Weitere Größen
    recess_depth: float = field(init=False)
    a_ein: float = field(init=False)                # Fläche eines tangentialen Einlasses in die Swirl Kammer
    a_ox: float = field(init=False)                 # Austrittsfläche Oxidator-Ringspalt

    # Oxidator Einlass Größen
    n_ox_ein: int                                   # Anzahl Oxidator Einlässe in den Ringspalt
    d_ox_ein: float                                 # Durchmesser Oxidator Einlässe
    a_ox_ein: float = field(init=False)             # Fläche eines Oxidator Einlasses


    def __post_init__(self):
        self.recess_depth = (self.d_ox - self.d_aus) / (2 * tan(deg2rad(self.phi)/2))
        self.a_ein = self.d_ein **2 / 4 * pi
        self.a_ox = (self.d_ox ** 2 - (self.d_aus + 2 * self.post_wall_thickness)**2) / 4 * pi
        self.a_ox_ein = self.d_ox_ein ** 2 / 4 * pi


    def __str__(self) -> str:
        return f"""
        LCSC Injector Geometry
        ----------------------------------------------------------------------------------------------
        Liquid Center: Fuel / Ethanol
        ----------------------------------------------------------------------------------------------
        d_ein / Durchmesser Einlass: {self.d_ein * 1e3: 14.2f} mm     n_ein / Anzahl Einlässe: {self.n_ein: 8.0f}
        d_aus / Durchmesser Auslass: {self.d_aus * 1e3: 14.2f} mm     n_ein / Länge Zyl. Auslass: {self.l_aus * 1e3: 8.2f} mm
        d_sk / Durchmesser Swirl Kammer: {self.d_sk * 1e3: 10.2f} mm     l_sk / Länge Swirl Kammer: {self.l_sk * 1e3: 9.2f} mm
        
        ----------------------------------------------------------------------------------------------
        Gas outer flow: Oxidizer / N2O
        ----------------------------------------------------------------------------------------------
        d_ox / Äußerer Durchmesser Ox: {self.d_ox * 1e3: 12.2f} mm
        Phi: {self.phi: 35.0f}      °      Recess Tiefe: {self.recess_depth * 1e3: 22.2f} mm
        d_ox_ein / Durchmesser Ox Einlass: {self.d_ox_ein * 1e3: 8.2f} mm     n_ox_ein / Anzahl Ox Einlässe: {self.n_ox_ein: 2.0f}
        ----------------------------------------------------------------------------------------------
        """





def size_lcsc_injector(
                    p_cc : float,
                    m_dot_fuel: float,
                    m_dot_ox: float,
                    n_fuel_ports: int,
                    n_ox_ports: int,
                    alpha: float,
                    v_ox: float,
                    post_wall_thickness: float,
                    oxidizer: str = 'N2O',
                    p_ox_inlet: float = None,
                    t_ox: float = None,
                    state_ox_inlet: AbstractState = None,
                    fuel: str = 'Ethanol',
                    p_fuel_inlet: float = None,
                    t_fuel: float = None,
                    state_fuel_inlet: AbstractState = None,
                       ):

    """
    Sizes a liquid centred swirl coaxial injector based on MA vom Schemm.
    Expects massflow rates for a single injector.
    Fluid states are considered at two different positions:
        "inlet" refers to the position upstream of the injector.
        "cc" refers to the position at the outlet of the injector into the combustion chamber.

    Fuel density is considered at inlet pressure as this is relevant for the tangential ports into the swirl chamber
    Oxidizer density is considered at combustion chamber pressure as this is relevant for the annular exit area.
    Oxidizer density in the chamber is calculated as isentropic flow from inlet conditions to chamber pressure.

    :param state_ox_inlet:                          CoolProp Abstract State
    :param state_fuel_inlet:                        CoolProp Abstract State
    :param p_cc:                    Pa              Combustion chamber pressure
    :param p_ox_inlet:              Pa              Oxidizer inlet pressure
    :param p_fuel_inlet:            Pa              Fuel inlet pressure
    :param t_fuel:                  Kelvin          Temperature fuel
    :param t_ox:                    Kelvin          Temperature ox
    :param m_dot_fuel:              kg/s            Fuel massflow for one injector
    :param m_dot_ox:                kg/s            Oxidizer massflow for one injector
    :param n_fuel_ports:                            Number of fuel ports into swirl chamber
    :param alpha:                   Degrees         Half spray angle
    :param v_ox:                    m/s             Oxidizer injection speed
    :param post_wall_thickness:     m               Thickness of annular wall between fuel and oxidizer
    :param oxidizer:                                CoolProp fluid name. Standard "N2O"
    :param fuel:                                    CoolProp fluid name. Standard "Ethanol"
    :return: LCSCInjector Object
    """

    # Bestimmung Dichte -----------------------------------------

    # Fuel ---------------------------------
    rho_fuel_inlet = calculate_density(fluid=fuel,
                                       state=state_fuel_inlet,
                                       pressure=p_fuel_inlet,
                                       temperature=t_fuel)

    print(f"Density Fuel Inlet: {rho_fuel_inlet:16.2f} kg/m^3")


    # Oxidizer ------------------------------
    rho_ox_inlet = calculate_density(fluid=oxidizer,
                                     state=state_ox_inlet,
                                     pressure=p_ox_inlet,
                                     temperature=t_ox)

    print(f"Density Oxidizer Inlet: {rho_ox_inlet:12.2f} kg/m^3")

    if state_ox_inlet is not None:
        entropy_ox_inlet = state_ox_inlet.smass()
    elif p_ox_inlet is not None and t_ox is not None:
        entropy_ox_inlet = PropsSI('S', 'P', p_ox_inlet, 'T', t_ox, oxidizer)

    state_ox_cc = AbstractState('REFPROP', oxidizer)
    state_ox_cc.update(PSmass_INPUTS, p_cc, entropy_ox_inlet)

    rho_ox_cc = state_ox_cc.rhomass()
    print(f"Density Oxidizer Chamber: {rho_ox_cc:10.2f} kg/m^3")



    # Swirl Kammer Geometrie ------------------------------------

    delta_p = p_fuel_inlet - p_cc

    # Auslass. Größe bei Schemm: d_aus
    X = get_uppercase_x(alpha=alpha)
    cd_aus = get_c_d_maximum_flow(X)
    A_aus = m_dot_fuel / (cd_aus * sqrt(2 * rho_fuel_inlet * delta_p))
    d_aus = sqrt(4 * A_aus / pi)

    print(f"Cd Outlet: {cd_aus:.4f}")
    print(f"A aus: {A_aus*1e6:.4f}")


    # Tangentiale Einlässe
    cd_ein = sqrt(X ** 3 / (2-X))
    print(f"Cd Inlet: {cd_ein:.4f}")
    A_ein = m_dot_fuel / (cd_ein * sqrt(2 * rho_fuel_inlet * delta_p))
    print(f"A ein: {A_ein*1e6:.4f}")
    d_ein = sqrt(4 * A_ein / (pi * n_fuel_ports))
    p_sc = p_fuel_inlet - ( m_dot_fuel/(cd_ein*A_ein)**2 / (2*rho_fuel_inlet) )
    dp_temp = ( m_dot_fuel/(cd_aus*A_aus))**2 / (2*rho_fuel_inlet)
    print(dp_temp*1e-5)
    print(f"Swirl chamber pressure should be around: {p_sc*1e-5:.5f} bar")


    if d_ein < 0.7 * 1e-3:
        print('WARNING: Orifice smaller than 0.7 mm might not be printable!')

    # Geometrie der Kammer
    d_sk = 3.3 * d_aus
    l_sk = d_sk
    l_aus = d_aus / 2

    # Combined cd value of fuel
    cd_A_equiv = 1 / sqrt( 1/(cd_ein*A_ein)**2 + 1/(cd_aus*A_aus)**2 ) # Equivalent Cd*A of Swirl Chamber
    print(f"Equivalent C_d with inlet area reference: {cd_A_equiv/A_ein:.4f}")


    # Oxidator Ringspalt Geometrie --------------------------------------
    A_ox = m_dot_ox / (v_ox * rho_ox_cc)
    d_ox = sqrt(4 * A_ox / pi + (d_aus + 2 * post_wall_thickness)**2)  # Subtract swirl chamberoutlet
    phi = 2 * alpha

    # Oxidator Einlässe in den Ringspalt
    heat_capacity_ratio_ox = get_heat_capacity_ratio(fluid=oxidizer,pressure=p_ox_inlet, temperature=t_ox, state=state_ox_inlet)
    gas_constant_ox = get_specific_gas_constant(fluid=oxidizer)
    A_ox_ein = m_dot_ox / (sqrt( heat_capacity_ratio_ox * gas_constant_ox * t_ox) * rho_ox_inlet)
    d_ox_ein = sqrt(4 * A_ox_ein / (pi * n_ox_ports))

    injector = LCSCInjector(
        d_ein=d_ein,
        n_ein=n_fuel_ports,
        d_aus=d_aus,
        l_aus=l_aus,
        d_ox=d_ox,
        d_sk=d_sk,
        l_sk=l_sk,
        phi=phi,
        post_wall_thickness=post_wall_thickness,
        n_ox_ein=n_ox_ports,
        d_ox_ein=d_ox_ein,
    )

    return injector




if __name__ == '__main__':



    r_inlet = 0.4 * 1e-3                # Einheit m. Moritz MA index: ein
    r_outlet = 1.64 * 1e-3              # Einheit m. Moritz MA index: aus
    n_injectors = 3

    injector = size_lcsc_injector(
            p_cc = 25 * 1e5,
            m_dot_fuel= 0.2/n_injectors,
            m_dot_ox= 0.8/n_injectors,
            n_fuel_ports= 3,
            n_ox_ports=6,
            alpha = 60,
            v_ox = 100,
            post_wall_thickness= 0.5 * 1e-3,
            oxidizer = 'N2O',
            p_ox_inlet = 40 * 1e5,
            t_ox= 500,
            fuel = 'Ethanol',
            p_fuel_inlet = 40 * 1e5,
            t_fuel= 300,
            )


    print(injector)
