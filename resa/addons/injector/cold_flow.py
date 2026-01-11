"""
Cold Flow Test Calculator for Swirl Injectors.

Scales hot-fire injector design to water/nitrogen testing conditions.
"""
import numpy as np

from .config import ColdFlowConfig, GeometryConfig
from .results import (
    InjectorGeometry, PerformanceMetrics, InjectorResults, ColdFlowResults
)
from .thermodynamics import (
    ThermodynamicCalculator, SprayAngleCorrelations, FilmThicknessCorrelations,
    calculate_swirl_number, calculate_choked_mass_flow
)


class ColdFlowCalculator:
    """
    Calculator for cold flow test equivalent injector sizing.

    Scales hot-fire injector design to water/nitrogen testing conditions.
    """

    def __init__(
        self,
        hot_fire_results: InjectorResults,
        cold_flow_config: ColdFlowConfig,
        geometry_config: GeometryConfig
    ):
        """
        Initialize cold flow calculator.

        Args:
            hot_fire_results: Results from hot-fire injector sizing
            cold_flow_config: Cold flow test conditions
            geometry_config: Injector geometry configuration
        """
        self.hot_results = hot_fire_results
        self.cold_config = cold_flow_config
        self.geom_config = geometry_config

    def calculate(self, is_gcsc: bool = True) -> ColdFlowResults:
        """
        Calculate cold flow equivalent injector parameters.

        Args:
            is_gcsc: Whether to use GCSC (True) or LCSC (False) configuration

        Returns:
            ColdFlowResults with equivalent cold flow parameters
        """
        T_amb = self.cold_config.ambient_temperature
        p_in = self.cold_config.inlet_pressure
        del_p = self.cold_config.pressure_drop

        # Get fluid properties for water and nitrogen
        water_inj = ThermodynamicCalculator.get_fluid_properties(
            'Water', T_amb, p_in, include_surface_tension=True
        )
        n2_inj = ThermodynamicCalculator.get_fluid_properties(
            'Nitrogen', T_amb, p_in
        )
        water_amb = ThermodynamicCalculator.get_fluid_properties(
            'Water', T_amb, p_in - del_p, include_surface_tension=True
        )
        n2_amb = ThermodynamicCalculator.get_fluid_properties(
            'Nitrogen', T_amb, p_in - del_p
        )

        # Get geometry from hot-fire results (use new naming with backward compat)
        r_fuel_orifice = self.hot_results.geometry.fuel_orifice_radius
        r_fuel_port = self.hot_results.geometry.fuel_port_radius
        r_sc = self.hot_results.geometry.swirl_chamber_radius
        r_ox_outlet = self.hot_results.geometry.ox_outlet_radius
        C_D = self.hot_results.performance.discharge_coefficient
        J = self.hot_results.performance.momentum_flux_ratio
        n_fuel_ports = self.geom_config.num_fuel_ports
        t_post = self.geom_config.post_thickness

        # Calculate water mass flow
        A_fuel_orifice = np.pi * r_fuel_orifice ** 2
        m_dot_w = C_D * A_fuel_orifice * np.sqrt(2 * water_inj.density * del_p)

        # Calculate film thickness
        if is_gcsc:
            t_film = FilmThicknessCorrelations.fu(
                m_dot_w, water_amb.viscosity, water_amb.density, del_p, r_sc
            )
        else:
            t_film = FilmThicknessCorrelations.suyari_lefebvre(
                m_dot_w, water_amb.viscosity, water_amb.density, del_p, r_fuel_orifice
            )

        r_aircore = r_fuel_orifice - t_film
        v_w_ax = m_dot_w / (water_amb.density * np.pi * (r_fuel_orifice ** 2 - r_aircore ** 2))

        # Calculate nitrogen velocity to match momentum flux ratio
        v_n2 = np.sqrt(J * v_w_ax ** 2 * water_amb.density / n2_amb.density)
        r_n2_outlet = r_ox_outlet

        # Calculate nitrogen mass flow
        if is_gcsc:
            m_dot_n2 = n2_amb.density * v_n2 * np.pi * r_n2_outlet ** 2
            alpha = SprayAngleCorrelations.anand(r_sc, r_fuel_port, n_fuel_ports, 1.5)
        else:
            m_dot_n2 = n2_amb.density * v_n2 * np.pi * (r_n2_outlet - t_post - r_fuel_orifice) ** 2
            X = r_aircore ** 2 / r_fuel_orifice ** 2
            alpha = SprayAngleCorrelations.lefebvre(X)

        # Calculate nitrogen inlet orifice (choked flow)
        A_n2_inlet_orifice = m_dot_n2 / calculate_choked_mass_flow(
            p_in, T_amb, 1.0, n2_amb.gamma, n2_inj.molar_mass
        )
        r_n2_inlet_orifice = np.sqrt(A_n2_inlet_orifice / np.pi)

        # Calculate metrics
        SN = calculate_swirl_number(r_sc, r_fuel_port, n_fuel_ports)
        sigma_w = water_amb.surface_tension or 0.072
        We = (n2_amb.density * ((v_n2 - v_w_ax) ** 2) * t_film) / sigma_w

        # Get recess length from hot-fire results
        l_recess = self.hot_results.geometry.recess_length

        geometry = InjectorGeometry(
            fuel_orifice_radius=r_fuel_orifice,
            fuel_port_radius=r_fuel_port,
            swirl_chamber_radius=r_sc,
            ox_outlet_radius=r_n2_outlet,
            ox_inlet_orifice_radius=r_n2_inlet_orifice,
            recess_length=l_recess
        )

        performance = PerformanceMetrics(
            spray_half_angle=np.rad2deg(alpha),
            swirl_number=SN,
            momentum_flux_ratio=J,
            velocity_ratio=v_n2 / v_w_ax,
            weber_number=We,
            discharge_coefficient=C_D,
            film_thickness=t_film
        )

        return ColdFlowResults(
            geometry=geometry,
            performance=performance,
            liquid_mass_flow=m_dot_w,
            gas_mass_flow=m_dot_n2,
            gas_velocity=v_n2,
            liquid_density=water_amb.density,
            gas_density=n2_amb.density
        )
