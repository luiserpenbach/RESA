"""
Liquid-Centered Swirl Coaxial (LCSC) Injector Calculator.

References:
    - Nardi, Rene et al. (2014): "Dimensioning a Simplex Swirl Injector"
      50th AIAA/ASME/SAE/ASEE Joint Propulsion Conference.
"""
import numpy as np

from .config import InjectorConfig
from .results import (
    InjectorGeometry, PerformanceMetrics, MassFlowResults, InjectorResults
)
from .thermodynamics import (
    ThermodynamicCalculator, DischargeCoefficients,
    calculate_swirl_number, calculate_open_area_ratio, calculate_choked_mass_flow
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
