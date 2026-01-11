"""
Thermodynamic property calculations for swirl injector design.

Uses CoolProp for fluid property calculations.
Falls back to approximate values if CoolProp is not available.
"""
import numpy as np
from typing import Optional

try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False

from .results import FluidProperties, PropellantProperties


# Fallback property data for common fluids (approximate values at 300K, 25bar)
FALLBACK_PROPERTIES = {
    'ethanol': {
        'density': 780.0,  # kg/m3
        'viscosity': 1.1e-3,  # Pa.s
        'surface_tension': 0.022,  # N/m
        'molar_mass': 0.04607,  # kg/mol
        'gamma': 1.13,
    },
    'nitrousoxide': {
        'density': 45.0,  # kg/m3 (gas at high temp)
        'viscosity': 1.5e-5,  # Pa.s
        'surface_tension': None,
        'molar_mass': 0.04401,  # kg/mol
        'gamma': 1.27,
    },
    'water': {
        'density': 998.0,  # kg/m3
        'viscosity': 1.0e-3,  # Pa.s
        'surface_tension': 0.072,  # N/m
        'molar_mass': 0.01802,  # kg/mol
        'gamma': 1.33,
    },
    'nitrogen': {
        'density': 28.0,  # kg/m3 (at ~25 bar)
        'viscosity': 1.8e-5,  # Pa.s
        'surface_tension': None,
        'molar_mass': 0.02801,  # kg/mol
        'gamma': 1.40,
    },
}


def _normalize_fluid_name(fluid: str) -> str:
    """Normalize fluid name for lookup."""
    # Remove REFPROP:: prefix if present
    if '::' in fluid:
        fluid = fluid.split('::')[1]
    return fluid.lower().replace(' ', '').replace('_', '')


class ThermodynamicCalculator:
    """Calculator for thermodynamic fluid properties using CoolProp."""

    @staticmethod
    def get_fluid_properties(
        fluid: str,
        temperature: float,
        pressure: float,
        include_surface_tension: bool = False
    ) -> FluidProperties:
        """
        Calculate fluid properties at given conditions.

        Args:
            fluid: CoolProp fluid name (e.g., 'REFPROP::Ethanol')
            temperature: Temperature in K
            pressure: Pressure in Pa
            include_surface_tension: Whether to calculate surface tension

        Returns:
            FluidProperties dataclass with calculated values
        """
        if COOLPROP_AVAILABLE:
            try:
                # Try with original fluid name
                density = CP.PropsSI('D', 'T', temperature, 'P', pressure, fluid)
                viscosity = CP.PropsSI('V', 'T', temperature, 'P', pressure, fluid)
                molar_mass = CP.PropsSI('M', fluid)
                gamma = CP.PropsSI('isentropic_expansion_coefficient', 'T', temperature, 'P', pressure, fluid)

                surface_tension = None
                if include_surface_tension:
                    try:
                        surface_tension = CP.PropsSI('surface_tension', 'T', temperature, 'P', pressure, fluid)
                    except Exception:
                        pass  # Surface tension not available for all fluids

                return FluidProperties(
                    density=density,
                    viscosity=viscosity,
                    surface_tension=surface_tension,
                    molar_mass=molar_mass,
                    gamma=gamma
                )
            except Exception:
                pass  # Fall through to fallback

        # Use fallback properties
        fluid_key = _normalize_fluid_name(fluid)
        if fluid_key not in FALLBACK_PROPERTIES:
            raise ValueError(f"Unknown fluid: {fluid}. Available: {list(FALLBACK_PROPERTIES.keys())}")

        props = FALLBACK_PROPERTIES[fluid_key]

        # Simple pressure/temperature corrections
        T_ref = 300.0
        P_ref = 25e5

        # Approximate density correction (ideal gas for gases, incompressible for liquids)
        if props['density'] < 100:  # Gas
            density = props['density'] * (pressure / P_ref) * (T_ref / temperature)
        else:  # Liquid
            density = props['density'] * (1 - 0.0005 * (temperature - T_ref))

        # Approximate viscosity correction
        viscosity = props['viscosity'] * (T_ref / temperature) ** 0.7

        return FluidProperties(
            density=density,
            viscosity=viscosity,
            surface_tension=props['surface_tension'] if include_surface_tension else None,
            molar_mass=props['molar_mass'],
            gamma=props['gamma']
        )

    @staticmethod
    def get_surface_tension_mixture(
        temperature: float,
        pressure: float,
        components: list[str],
        mass_fractions: list[float]
    ) -> float:
        """
        Estimate surface tension for a mixture using mass-weighted average.

        Args:
            temperature: Temperature in K
            pressure: Pressure in Pa
            components: List of component names
            mass_fractions: Mass fractions of each component

        Returns:
            Estimated surface tension in N/m
        """
        sigma_total = 0.0
        for comp, w in zip(components, mass_fractions):
            try:
                if COOLPROP_AVAILABLE:
                    sigma = CP.PropsSI('surface_tension', 'T', temperature, 'P', pressure, comp)
                    sigma_total += w * sigma
                else:
                    raise ValueError("CoolProp not available")
            except Exception:
                # Fallback values for common fluids
                fallback = {
                    'water': 0.072,
                    'ethanol': 0.022,
                }
                sigma_total += w * fallback.get(comp.lower(), 0.03)
        return sigma_total

    @classmethod
    def get_propellant_properties(
        cls,
        fuel: str,
        oxidizer: str,
        fuel_temperature: float,
        oxidizer_temperature: float,
        inlet_pressure: float,
        chamber_pressure: float
    ) -> PropellantProperties:
        """
        Calculate complete propellant properties at inlet and chamber conditions.

        Args:
            fuel: CoolProp fuel fluid name
            oxidizer: CoolProp oxidizer fluid name
            fuel_temperature: Fuel temperature in K
            oxidizer_temperature: Oxidizer temperature in K
            inlet_pressure: Injector inlet pressure in Pa
            chamber_pressure: Combustion chamber pressure in Pa

        Returns:
            PropellantProperties with all calculated values
        """
        # Properties at inlet
        fuel_inlet = cls.get_fluid_properties(
            fuel, fuel_temperature, inlet_pressure, include_surface_tension=True
        )
        oxidizer_inlet = cls.get_fluid_properties(
            oxidizer, oxidizer_temperature, inlet_pressure
        )

        # Properties at chamber
        fuel_chamber = cls.get_fluid_properties(
            fuel, fuel_temperature, chamber_pressure, include_surface_tension=True
        )
        oxidizer_chamber = cls.get_fluid_properties(
            oxidizer, oxidizer_temperature, chamber_pressure
        )

        # Estimate surface tension for ethanol-water mixtures
        if 'ethanol' in fuel.lower():
            fuel_chamber = FluidProperties(
                density=fuel_chamber.density,
                viscosity=fuel_chamber.viscosity,
                surface_tension=cls.get_surface_tension_mixture(
                    fuel_temperature, chamber_pressure,
                    ['Ethanol', 'Water'], [0.8, 0.2]
                ),
                molar_mass=fuel_chamber.molar_mass,
                gamma=fuel_chamber.gamma
            )

        return PropellantProperties(
            fuel_at_inlet=fuel_inlet,
            oxidizer_at_inlet=oxidizer_inlet,
            fuel_at_chamber=fuel_chamber,
            oxidizer_at_chamber=oxidizer_chamber
        )


class DischargeCoefficients:
    """Collection of discharge coefficient correlations."""

    @staticmethod
    def maximum_flow(X: float) -> float:
        """
        Discharge coefficient assuming maximum flow condition.

        Args:
            X: Open area ratio (A_aircore / A_exit)

        Returns:
            Discharge coefficient
        """
        return np.sqrt(((1 - X) ** 3) / (1 + X))

    @staticmethod
    def abramovic(r_sc: float, r_p: float, n_p: int, r_o: Optional[float] = None) -> float:
        """
        Discharge coefficient from Abramovic correlation.

        Args:
            r_sc: Swirl chamber radius in m
            r_p: Port radius in m
            n_p: Number of tangential ports
            r_o: Orifice radius (defaults to r_sc)

        Returns:
            Discharge coefficient
        """
        if r_o is None:
            r_o = r_sc
        A = (r_sc - r_p) * r_o / (n_p * r_p ** 2)
        return 0.432 * A ** -0.64

    @staticmethod
    def rizk_lefebvre(r_sc: float, r_p: float, n_p: int, r_o: Optional[float] = None) -> float:
        """
        Discharge coefficient from Rizk and Lefebvre (1985).

        Args:
            r_sc: Swirl chamber radius in m
            r_p: Port radius in m
            n_p: Number of tangential ports
            r_o: Orifice radius (defaults to r_sc)

        Returns:
            Discharge coefficient
        """
        if r_o is None:
            r_o = r_sc
        A_tot_p = n_p * np.pi * r_p ** 2
        return 0.35 * np.sqrt(A_tot_p / (4 * r_sc * r_o)) * (r_sc / r_o) ** 0.25

    @staticmethod
    def hong(r_sc: float, r_p: float, n_p: int, r_o: Optional[float] = None) -> float:
        """
        Discharge coefficient from Hong et al.

        Args:
            r_sc: Swirl chamber radius in m
            r_p: Port radius in m
            n_p: Number of tangential ports
            r_o: Orifice radius (defaults to r_sc)

        Returns:
            Discharge coefficient
        """
        if r_o is None:
            r_o = r_sc
        beta = (r_sc - r_p) / r_o
        A_tot_p = n_p * np.pi * r_p ** 2
        return 0.44 * (A_tot_p / (4 * r_o ** 2)) ** (0.84 * beta ** -0.52) * beta ** -0.59

    @staticmethod
    def fu(r_sc: float, r_p: float, n_p: int) -> float:
        """
        Discharge coefficient from Fu et al.

        Args:
            r_sc: Swirl chamber radius in m
            r_p: Port radius in m
            n_p: Number of tangential ports

        Returns:
            Discharge coefficient
        """
        A = (r_sc - r_p) * r_sc / (n_p * r_p ** 2)
        return 0.4354 * A ** -0.877

    @staticmethod
    def anand(r_sc: float, r_p: float, n_p: int) -> float:
        """
        Discharge coefficient from Anand et al.

        Args:
            r_sc: Swirl chamber radius in m
            r_p: Port radius in m
            n_p: Number of tangential ports

        Returns:
            Discharge coefficient
        """
        A = (r_sc - r_p) * r_sc / (n_p * r_p ** 2)
        return 1.28 * A ** -1.28


class SprayAngleCorrelations:
    """Collection of spray angle correlations."""

    @staticmethod
    def lefebvre(X: float) -> float:
        """
        Spray half angle from Lefebvre correlation.

        Args:
            X: Open area ratio

        Returns:
            Spray half angle in radians
        """
        return np.arcsin(X * np.sqrt(8) / (1 + np.sqrt(X) * np.sqrt(1 + X)))

    @staticmethod
    def anand(r_sc: float, r_p: float, n_p: int, RR: float = 1.5) -> float:
        """
        Spray half angle from Anand et al.

        Args:
            r_sc: Swirl chamber radius in m
            r_p: Port radius in m
            n_p: Number of tangential ports
            RR: Recess ratio (default 1.5)

        Returns:
            Spray half angle in radians
        """
        A = (r_sc - r_p) * r_sc / (n_p * r_p ** 2)
        return np.arctan(0.01 * A ** 1.64 * RR ** -0.242)

    @staticmethod
    def fu(m_dot_f: float, eta_f: float, r_sc: float, r_p: float,
           n_p: int, r_o: Optional[float] = None) -> float:
        """
        Spray half angle from Fu et al.

        Args:
            m_dot_f: Fuel mass flow in kg/s
            eta_f: Fuel viscosity in Pa.s
            r_sc: Swirl chamber radius in m
            r_p: Port radius in m
            n_p: Number of tangential ports
            r_o: Orifice radius (defaults to r_sc)

        Returns:
            Spray half angle in radians
        """
        if r_o is None:
            r_o = r_sc
        Re_p = 2 * m_dot_f / (np.pi * r_p * eta_f * np.sqrt(n_p))
        A = (r_sc - r_p) * r_o / (n_p * r_p ** 2)
        return np.arctan(0.033 * A ** 0.338 * Re_p ** 0.249)


class FilmThicknessCorrelations:
    """Collection of film thickness correlations."""

    @staticmethod
    def fu(m_dot_f: float, eta_f: float, rho_f: float,
           del_p: float, r_sc: float) -> float:
        """
        Film thickness from Fu et al. (based on Rizk and Lefebvre).

        For open-end swirl injectors.

        Args:
            m_dot_f: Fuel mass flow in kg/s
            eta_f: Fuel viscosity in Pa.s
            rho_f: Fuel density in kg/m3
            del_p: Pressure drop in Pa
            r_sc: Swirl chamber radius in m

        Returns:
            Film thickness in m
        """
        return 3.1 * ((2 * r_sc * m_dot_f * eta_f) / (rho_f * del_p)) ** 0.25

    @staticmethod
    def suyari_lefebvre(m_dot_f: float, eta_f: float, rho_f: float,
                        del_p: float, r_o: float) -> float:
        """
        Film thickness from Suyari and Lefebvre.

        Args:
            m_dot_f: Fuel mass flow in kg/s
            eta_f: Fuel viscosity in Pa.s
            rho_f: Fuel density in kg/m3
            del_p: Pressure drop in Pa
            r_o: Orifice radius in m

        Returns:
            Film thickness in m
        """
        return 2.7 * ((2 * r_o * m_dot_f * eta_f) / (rho_f * del_p)) ** 0.25

    @staticmethod
    def simmons_harding(m_dot_f: float, alpha: float, rho_f: float,
                        del_p: float, r_sc: float) -> float:
        """
        Film thickness from Simmons and Harding.

        Args:
            m_dot_f: Fuel mass flow in kg/s
            alpha: Spray half angle in degrees
            rho_f: Fuel density in kg/m3
            del_p: Pressure drop in Pa
            r_sc: Swirl chamber radius in m

        Returns:
            Film thickness in m
        """
        return (0.00805 * np.sqrt(rho_f) * m_dot_f) / (
            np.sqrt(del_p * rho_f) * 2 * r_sc * np.cos(np.deg2rad(alpha))
        )


def calculate_swirl_number(r_sc: float, r_p: float, n_p: int, r_o: Optional[float] = None) -> float:
    """
    Calculate geometric swirl number.

    Args:
        r_sc: Swirl chamber radius in m
        r_p: Port radius in m
        n_p: Number of tangential ports
        r_o: Orifice radius (defaults to r_sc)

    Returns:
        Swirl number (dimensionless)
    """
    if r_o is None:
        r_o = r_sc
    return (r_sc - r_p) * r_o / (n_p * r_p ** 2)


def calculate_open_area_ratio(alpha_deg: float) -> float:
    """
    Calculate open area ratio from spray half angle.

    Uses empirical correlation: X = 0.0042 * alpha^1.2714

    Args:
        alpha_deg: Spray half angle in degrees

    Returns:
        Open area ratio X = A_aircore / A_exit
    """
    return 0.0042 * alpha_deg ** 1.2714


def calculate_choked_mass_flow(
    pressure: float,
    temperature: float,
    area: float,
    gamma: float,
    molar_mass: float
) -> float:
    """
    Calculate choked mass flow through an orifice.

    Args:
        pressure: Upstream pressure in Pa
        temperature: Upstream temperature in K
        area: Orifice area in m2
        gamma: Specific heat ratio
        molar_mass: Molar mass in kg/mol

    Returns:
        Mass flow rate in kg/s
    """
    R = 8.314  # Universal gas constant
    return (
        area * pressure / np.sqrt(temperature) *
        np.sqrt(gamma * molar_mass / R) *
        ((gamma + 1) / 2) ** ((-gamma + 1) / (2 * (gamma - 1)))
    )
