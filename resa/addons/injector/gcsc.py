"""
Gas-Centered Swirl Coaxial (GCSC) Injector Calculator.

References:
    - Anand et al. correlations for gas-centered configurations.
"""
import numpy as np

from .config import InjectorConfig
from .results import (
    InjectorGeometry, PerformanceMetrics, MassFlowResults, InjectorResults
)
from .thermodynamics import (
    ThermodynamicCalculator, DischargeCoefficients, SprayAngleCorrelations,
    FilmThicknessCorrelations, calculate_swirl_number, calculate_choked_mass_flow
)


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
