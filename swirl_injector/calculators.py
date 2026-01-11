"""
Swirl Injector Dimensioning Calculators.

References:
    - LCSC: Nardi, Rene et al. (2014): "Dimensioning a Simplex Swirl Injector"
      50th AIAA/ASME/SAE/ASEE Joint Propulsion Conference.
    - GCSC: Anand et al. correlations for gas-centered configurations.
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from config import InjectorConfig, ColdFlowConfig, GeometryConfig
from results import (
    InjectorGeometry, PerformanceMetrics, MassFlowResults,
    InjectorResults, ColdFlowResults, FluidProperties
)
from thermodynamics import (
    ThermodynamicCalculator, DischargeCoefficients, SprayAngleCorrelations,
    FilmThicknessCorrelations, calculate_swirl_number, calculate_open_area_ratio,
    calculate_choked_mass_flow
)


class LCSCCalculator:
    """
    Liquid-Centered Swirl Coaxial (LCSC) injector calculator.
    
    Uses correlations from Rezende et al. for sizing.
    """
    
    def __init__(self, config: InjectorConfig):
        """
        Initialize calculator with configuration.
        
        Args:
            config: Complete injector configuration
        """
        self.config = config
        self._props = None
    
    @property
    def propellant_properties(self):
        """Lazily calculate propellant properties."""
        if self._props is None:
            self._props = ThermodynamicCalculator.get_propellant_properties(
                fuel=self.config.propellants.fuel,
                oxidizer=self.config.propellants.oxidizer,
                fuel_temperature=self.config.propellants.fuel_temperature,
                oxidizer_temperature=self.config.propellants.oxidizer_temperature,
                inlet_pressure=self.config.operating.inlet_pressure,
                chamber_pressure=self.config.operating.chamber_pressure
            )
        return self._props
    
    def calculate(self) -> InjectorResults:
        """
        Calculate LCSC injector dimensions and performance.
        
        Returns:
            Complete InjectorResults with geometry, performance, and mass flows
        """
        props = self.propellant_properties
        op = self.config.operating
        geom = self.config.geometry
        
        # Get fluid properties shortcuts
        rho_f_inj = props.fuel_at_inlet.density
        rho_ox_inj = props.oxidizer_at_inlet.density
        eta_f_inj = props.fuel_at_inlet.viscosity
        
        rho_f_cc = props.fuel_at_chamber.density
        rho_ox_cc = props.oxidizer_at_chamber.density
        eta_f_cc = props.fuel_at_chamber.viscosity
        sigma_f_cc = props.fuel_at_chamber.surface_tension or 0.025
        
        gamma_ox = props.oxidizer_at_chamber.gamma
        M_ox = props.oxidizer_at_inlet.molar_mass
        
        # Calculate open area ratio from spray angle
        X = calculate_open_area_ratio(geom.spray_half_angle)
        
        # Calculate fuel orifice size using maximum flow assumption
        C_D = DischargeCoefficients.maximum_flow(X)
        A_tot_fuel_orifice = op.mass_flow_fuel / (C_D * np.sqrt(2 * rho_f_inj * op.pressure_drop))
        r_fuel_orifice = np.sqrt(4 * A_tot_fuel_orifice / (np.pi * geom.num_elements)) / 2

        # Calculate fuel port size (tangential inlets)
        C_D_p = np.sqrt(X ** 3 / (2 - X))  # tangential port discharge coefficient
        A_tot_fuel_port = op.mass_flow_fuel / (
            geom.num_elements * C_D_p * np.sqrt(2 * rho_f_inj * op.pressure_drop)
        )
        r_fuel_port = np.sqrt(A_tot_fuel_port / (np.pi * geom.num_fuel_ports))

        # Calculate swirl chamber geometry
        r_sc = 3.3 * r_fuel_orifice
        l_sc = 2 * r_sc
        l_fuel_orifice = r_fuel_orifice

        # Calculate oxidizer inlet orifice (choked flow) - depends on num_ox_orifices
        # Total area divided by number of orifices per element and number of elements
        A_ox_inlet_total = op.mass_flow_oxidizer / calculate_choked_mass_flow(
            op.inlet_pressure, self.config.propellants.oxidizer_temperature,
            1.0, gamma_ox, M_ox
        )
        A_ox_inlet_per_orifice = A_ox_inlet_total / (geom.num_elements * geom.num_ox_orifices)
        r_ox_inlet_orifice = np.sqrt(A_ox_inlet_per_orifice / np.pi)

        # Calculate oxidizer outlet size (sized by target velocity)
        A_ox_outlet = op.mass_flow_oxidizer / (
            op.oxidizer_velocity * rho_ox_cc * geom.num_elements
        )
        r_ox_outlet = np.sqrt(A_ox_outlet / np.pi + (r_fuel_orifice + geom.post_thickness) ** 2)

        # Calculate performance metrics
        SN = calculate_swirl_number(r_sc, r_fuel_port, geom.num_fuel_ports)
        r_aircore = X * r_fuel_orifice
        t_film = r_fuel_orifice * (1 - np.sqrt(X))

        v_f_ax = op.mass_flow_fuel / (
            geom.num_elements * rho_f_cc * np.pi * (r_fuel_orifice ** 2 - r_aircore ** 2)
        )

        J = (rho_ox_cc * op.oxidizer_velocity ** 2) / (rho_f_cc * v_f_ax ** 2)
        RV = op.oxidizer_velocity / v_f_ax
        We = (rho_ox_cc * ((op.oxidizer_velocity - v_f_ax) ** 2) * t_film) / sigma_f_cc
        Re_p = 2 * op.mass_flow_fuel / (np.pi * r_fuel_port * eta_f_inj * np.sqrt(geom.num_fuel_ports))

        # Calculate recess length for mixing
        # l_recess = (ox_outlet_radius - fuel_orifice_radius) / tan(alpha)
        # where alpha is spray half angle
        alpha_rad = np.deg2rad(geom.spray_half_angle)
        if alpha_rad > 0 and r_ox_outlet > r_fuel_orifice:
            l_recess = (r_ox_outlet - r_fuel_orifice) / np.tan(alpha_rad)
        else:
            l_recess = 0.0

        # Build results
        geometry = InjectorGeometry(
            fuel_orifice_radius=r_fuel_orifice,
            fuel_port_radius=r_fuel_port,
            swirl_chamber_radius=r_sc,
            ox_outlet_radius=r_ox_outlet,
            ox_inlet_orifice_radius=r_ox_inlet_orifice,
            recess_length=l_recess,
            fuel_orifice_length=l_fuel_orifice,
            swirl_chamber_length=l_sc
        )

        performance = PerformanceMetrics(
            spray_half_angle=geom.spray_half_angle,
            swirl_number=SN,
            momentum_flux_ratio=J,
            velocity_ratio=RV,
            weber_number=We,
            discharge_coefficient=C_D,
            reynolds_port=Re_p,
            film_thickness=t_film,
            aircore_radius=r_aircore
        )

        mass_flows = MassFlowResults(
            fuel_per_element=op.mass_flow_fuel / geom.num_elements,
            oxidizer_per_element=op.mass_flow_oxidizer / geom.num_elements,
            total_fuel=op.mass_flow_fuel,
            total_oxidizer=op.mass_flow_oxidizer
        )

        return InjectorResults(
            geometry=geometry,
            performance=performance,
            mass_flows=mass_flows,
            propellant_properties=props,
            injector_type="LCSC"
        )

    def compare_discharge_coefficients(self) -> dict:
        """
        Compare different discharge coefficient correlations.

        Returns:
            Dictionary with correlation names and values
        """
        results = self.calculate()
        r_sc = results.geometry.swirl_chamber_radius
        r_p = results.geometry.fuel_port_radius
        r_o = results.geometry.fuel_orifice_radius
        n_p = self.config.geometry.num_fuel_ports

        X = calculate_open_area_ratio(self.config.geometry.spray_half_angle)

        return {
            "maximum_flow": DischargeCoefficients.maximum_flow(X),
            "abramovic": DischargeCoefficients.abramovic(r_sc, r_p, n_p, r_o),
            "rizk_lefebvre": DischargeCoefficients.rizk_lefebvre(r_sc, r_p, n_p, r_o),
        }


class GCSCCalculator:
    """
    Gas-Centered Swirl Coaxial (GCSC) injector calculator.

    Uses correlations from Anand et al. for sizing.
    """

    def __init__(self, config: InjectorConfig):
        """
        Initialize calculator with configuration.

        Args:
            config: Complete injector configuration
        """
        self.config = config
        self._props = None

    @property
    def propellant_properties(self):
        """Lazily calculate propellant properties."""
        if self._props is None:
            self._props = ThermodynamicCalculator.get_propellant_properties(
                fuel=self.config.propellants.fuel,
                oxidizer=self.config.propellants.oxidizer,
                fuel_temperature=self.config.propellants.fuel_temperature,
                oxidizer_temperature=self.config.propellants.oxidizer_temperature,
                inlet_pressure=self.config.operating.inlet_pressure,
                chamber_pressure=self.config.operating.chamber_pressure
            )
        return self._props

    def calculate(self) -> InjectorResults:
        """
        Calculate GCSC injector dimensions and performance.

        Uses iterative approach to match target mass flow.

        Returns:
            Complete InjectorResults with geometry, performance, and mass flows
        """
        props = self.propellant_properties
        op = self.config.operating
        geom = self.config.geometry

        # Get fluid properties shortcuts
        rho_f_inj = props.fuel_at_inlet.density
        rho_ox_inj = props.oxidizer_at_inlet.density
        eta_f_inj = props.fuel_at_inlet.viscosity

        rho_f_cc = props.fuel_at_chamber.density
        rho_ox_cc = props.oxidizer_at_chamber.density
        eta_f_cc = props.fuel_at_chamber.viscosity
        sigma_f_cc = props.fuel_at_chamber.surface_tension or 0.025

        gamma_ox = props.oxidizer_at_chamber.gamma
        M_ox = props.oxidizer_at_inlet.molar_mass

        # Size oxidizer inlet orifice (choked flow) - depends on num_ox_orifices
        A_ox_inlet_total = op.mass_flow_oxidizer / calculate_choked_mass_flow(
            op.inlet_pressure, self.config.propellants.oxidizer_temperature,
            1.0, gamma_ox, M_ox
        )
        A_ox_inlet_per_orifice = A_ox_inlet_total / (geom.num_elements * geom.num_ox_orifices)
        r_ox_inlet_orifice = np.sqrt(A_ox_inlet_per_orifice / np.pi)

        # Size oxidizer outlet (by target velocity)
        A_ox_outlet = op.mass_flow_oxidizer / (op.oxidizer_velocity * rho_ox_cc * geom.num_elements)
        r_ox_outlet = np.sqrt(A_ox_outlet / np.pi)

        # Iterative sizing of fuel swirl injector to match mass flow
        r_fuel_port = 0.0004  # initial port radius
        m_dot_f_current = 0.0
        tolerance = 0.01
        max_iterations = 100

        for _ in range(max_iterations):
            r_fuel_orifice = r_ox_outlet + geom.post_thickness + geom.minimum_clearance
            t_film = FilmThicknessCorrelations.fu(
                op.mass_flow_fuel, eta_f_cc, rho_f_cc, op.pressure_drop, r_fuel_orifice
            )
            t_film = min(t_film, geom.minimum_clearance)

            A_fuel_orifice = np.pi * r_fuel_orifice ** 2
            C_D = DischargeCoefficients.anand(r_fuel_orifice, r_fuel_port, geom.num_fuel_ports)
            m_dot_element = C_D * A_fuel_orifice * np.sqrt(2 * rho_f_inj * op.pressure_drop)
            m_dot_f_current = geom.num_elements * m_dot_element

            if abs(m_dot_f_current - op.mass_flow_fuel) / op.mass_flow_fuel < tolerance:
                break

            r_fuel_port = r_fuel_port * np.sqrt(op.mass_flow_fuel / m_dot_f_current)

        # For GCSC, swirl chamber radius equals fuel orifice radius
        r_sc = r_fuel_orifice

        # Calculate performance metrics
        SN = calculate_swirl_number(r_fuel_orifice, r_fuel_port, geom.num_fuel_ports)
        r_aircore = r_fuel_orifice - t_film
        alpha_rad = SprayAngleCorrelations.anand(r_fuel_orifice, r_fuel_port, geom.num_fuel_ports, 1.5)

        v_f_ax = op.mass_flow_fuel / (
            geom.num_elements * rho_f_cc * np.pi * (r_fuel_orifice ** 2 - r_aircore ** 2)
        )

        J = (rho_ox_cc * op.oxidizer_velocity ** 2) / (rho_f_cc * v_f_ax ** 2)
        RV = op.oxidizer_velocity / v_f_ax
        We = (rho_ox_cc * ((op.oxidizer_velocity - v_f_ax) ** 2) * t_film) / sigma_f_cc
        Re_p = 2 * op.mass_flow_fuel / (np.pi * r_fuel_port * eta_f_inj * np.sqrt(geom.num_fuel_ports))

        # Calculate recess length for mixing
        # l_recess = (fuel_orifice_radius - ox_outlet_radius) / tan(alpha)
        # For GCSC, fuel is on outside, so we use fuel_orifice - ox_outlet
        if alpha_rad > 0 and r_fuel_orifice > r_ox_outlet:
            l_recess = (r_fuel_orifice - r_ox_outlet) / np.tan(alpha_rad)
        else:
            l_recess = 0.0

        # Build results
        geometry = InjectorGeometry(
            fuel_orifice_radius=r_fuel_orifice,
            fuel_port_radius=r_fuel_port,
            swirl_chamber_radius=r_sc,
            ox_outlet_radius=r_ox_outlet,
            ox_inlet_orifice_radius=r_ox_inlet_orifice,
            recess_length=l_recess
        )

        performance = PerformanceMetrics(
            spray_half_angle=np.rad2deg(alpha_rad),
            swirl_number=SN,
            momentum_flux_ratio=J,
            velocity_ratio=RV,
            weber_number=We,
            discharge_coefficient=C_D,
            reynolds_port=Re_p,
            film_thickness=t_film,
            aircore_radius=r_aircore
        )

        mass_flows = MassFlowResults(
            fuel_per_element=m_dot_element,
            oxidizer_per_element=op.mass_flow_oxidizer / geom.num_elements,
            total_fuel=m_dot_f_current,
            total_oxidizer=op.mass_flow_oxidizer
        )

        return InjectorResults(
            geometry=geometry,
            performance=performance,
            mass_flows=mass_flows,
            propellant_properties=props,
            injector_type="GCSC"
        )

    def compare_discharge_coefficients(self) -> dict:
        """
        Compare different discharge coefficient correlations.

        Returns:
            Dictionary with correlation names and values
        """
        results = self.calculate()
        r_sc = results.geometry.swirl_chamber_radius
        r_p = results.geometry.fuel_port_radius
        n_p = self.config.geometry.num_fuel_ports

        return {
            "anand": DischargeCoefficients.anand(r_sc, r_p, n_p),
            "fu": DischargeCoefficients.fu(r_sc, r_p, n_p),
            "hong": DischargeCoefficients.hong(r_sc, r_p, n_p),
        }


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