"""
Injector orifice sizing for torch igniter.

Implements proper HEM (Homogeneous Equilibrium Model) for N2O two-phase flow.
"""

import numpy as np
from typing import Dict, Tuple
import warnings

from .fluids import FluidProperties
from .nozzle import diameter_from_area


class InjectorDesigner:
    """Injector orifice sizing with proper two-phase flow modeling."""

    def __init__(self):
        """Initialize with fluid properties calculator."""
        self.fluids = FluidProperties()

    def size_ethanol_orifice(
        self,
        mass_flow: float,
        feed_pressure: float,
        chamber_pressure: float,
        orifice_count: int,
        discharge_coefficient: float,
        feed_temperature: float = 298.15
    ) -> Dict[str, float]:
        """Size ethanol orifice using incompressible flow.

        Args:
            mass_flow: Total ethanol mass flow rate (kg/s)
            feed_pressure: Feed pressure (Pa)
            chamber_pressure: Chamber pressure (Pa)
            orifice_count: Number of orifices
            discharge_coefficient: Cd (typically 0.6-0.8)
            feed_temperature: Feed temperature (K)

        Returns:
            dict: orifice_diameter, total_area, injection_velocity,
                  pressure_drop, mass_flux
        """
        rho = self.fluids.ethanol_density(feed_temperature, feed_pressure)
        dp = feed_pressure - chamber_pressure

        if dp <= 0:
            raise ValueError("Feed pressure must exceed chamber pressure")

        m_dot_per_orifice = mass_flow / orifice_count
        area_per_orifice = m_dot_per_orifice / (
            discharge_coefficient * np.sqrt(2 * rho * dp)
        )

        diameter = diameter_from_area(area_per_orifice)
        total_area = area_per_orifice * orifice_count
        velocity = np.sqrt(2 * dp / rho)
        mass_flux = m_dot_per_orifice / area_per_orifice

        return {
            'orifice_diameter': diameter,
            'total_area': total_area,
            'injection_velocity': velocity,
            'pressure_drop': dp,
            'mass_flux': mass_flux,
        }

    def size_n2o_orifice_hem(
        self,
        mass_flow: float,
        feed_pressure: float,
        chamber_pressure: float,
        orifice_count: int,
        discharge_coefficient: float,
        feed_temperature: float = 293.15
    ) -> Dict[str, float]:
        """Size N2O orifice using proper HEM (Homogeneous Equilibrium Model).

        This implementation properly accounts for:
        - Two-phase flashing during expansion
        - Critical (choked) flow conditions
        - Thermodynamic equilibrium (saturation maintained)
        - Homogeneous flow (liquid and vapor at same velocity)

        Method:
        1. Assumes liquid enters at feed conditions
        2. Integrates momentum equation through expansion
        3. Finds critical point where flow becomes sonic
        4. Calculates mass flux at critical conditions

        Args:
            mass_flow: Total N2O mass flow rate (kg/s)
            feed_pressure: Feed pressure (Pa)
            chamber_pressure: Chamber pressure (Pa)
            orifice_count: Number of orifices
            discharge_coefficient: Cd (typically 0.6-0.8)
            feed_temperature: Feed temperature (K)

        Returns:
            dict: orifice_diameter, total_area, injection_velocity,
                  pressure_drop, mass_flux, quality_exit, choked
        """
        # Get inlet saturated liquid properties
        try:
            inlet_sat = self.fluids.n2o_saturation_properties(feed_temperature)
        except Exception as e:
            raise ValueError(f"Failed to get N2O properties at {feed_temperature:.1f} K: {e}")

        P_sat = inlet_sat['P_sat']
        rho_inlet = inlet_sat['rho_liquid']
        h_inlet = inlet_sat['h_liquid']

        # Validate feed conditions
        if feed_pressure < P_sat * 0.95:
            warnings.warn(
                f"Feed pressure ({feed_pressure/1e5:.1f} bar) < saturation pressure "
                f"({P_sat/1e5:.1f} bar) at {feed_temperature:.1f} K. "
                "N2O may be two-phase in feed system."
            )

        # Check if feed is significantly above saturation (subcooled)
        if feed_pressure > P_sat * 1.2:
            # Subcooled liquid - use modified inlet enthalpy
            # For subcooled, h ~ h_sat + v_liquid * (P - P_sat)
            # For liquids, v is small so this correction is minor
            h_inlet_actual = h_inlet  # Simplification: ignore subcooling effect
        else:
            h_inlet_actual = h_inlet

        dp = feed_pressure - chamber_pressure
        if dp <= 0:
            raise ValueError("Feed pressure must exceed chamber pressure")

        # Determine if flow is choked
        is_choked = dp > 0.1 * feed_pressure  # Almost always true for N2O

        m_dot_per_orifice = mass_flow / orifice_count

        if is_choked:
            # Find critical conditions using proper HEM
            result = self._solve_hem_choked(
                P_inlet=feed_pressure,
                h_inlet=h_inlet_actual,
                m_dot=m_dot_per_orifice,
                Cd=discharge_coefficient,
                feed_temperature=feed_temperature
            )
        else:
            # Non-choked: expand to chamber pressure
            result = self._solve_hem_nonchoked(
                P_inlet=feed_pressure,
                P_exit=chamber_pressure,
                h_inlet=h_inlet_actual,
                m_dot=m_dot_per_orifice,
                Cd=discharge_coefficient,
                feed_temperature=feed_temperature
            )

        # Calculate geometry
        area_per_orifice = result['area']
        diameter = diameter_from_area(area_per_orifice)
        total_area = area_per_orifice * orifice_count

        # Validation warnings
        if result['quality'] > 0.3:
            warnings.warn(
                f"Exit quality = {result['quality']:.2f} is high (>0.3). "
                "HEM model accuracy may be reduced."
            )

        if diameter < 0.0005 or diameter > 0.01:
            warnings.warn(
                f"Orifice diameter = {diameter*1000:.2f} mm outside typical range (0.5-10 mm)."
            )

        return {
            'orifice_diameter': diameter,
            'total_area': total_area,
            'injection_velocity': result['velocity'],
            'pressure_drop': dp,
            'mass_flux': result['mass_flux'],
            'quality_exit': result['quality'],
            'choked': is_choked,
        }

    def _solve_hem_choked(
        self,
        P_inlet: float,
        h_inlet: float,
        m_dot: float,
        Cd: float,
        feed_temperature: float
    ) -> Dict[str, float]:
        """Solve for choked HEM flow conditions.

        For HEM choked flow, the critical point is where:
        dG/dP = 0 (maximum mass flux)

        This occurs when the flow reaches the frozen speed of sound
        for the two-phase mixture.

        Method: Iterate to find pressure where mass flux is maximum.

        Args:
            P_inlet: Inlet pressure (Pa)
            h_inlet: Inlet specific enthalpy (J/kg)
            m_dot: Mass flow rate per orifice (kg/s)
            Cd: Discharge coefficient
            feed_temperature: Feed temperature (K)

        Returns:
            dict with area, mass_flux, velocity, quality
        """
        # Critical pressure is typically 0.55-0.60 x P_inlet for saturated N2O
        # We'll search for the maximum mass flux

        # Pressure search range
        P_min = 0.4 * P_inlet
        P_max = 0.8 * P_inlet
        n_points = 30

        P_array = np.linspace(P_min, P_max, n_points)
        G_array = np.zeros(n_points)

        # Calculate mass flux at each pressure
        for i, P in enumerate(P_array):
            try:
                G_array[i] = self._calculate_mass_flux_hem(
                    P_inlet, P, h_inlet, feed_temperature
                )
            except:
                G_array[i] = 0.0

        # Find maximum
        idx_max = np.argmax(G_array)
        P_critical = P_array[idx_max]
        G_max = G_array[idx_max]

        # Refine around maximum using golden section search
        P_critical, G_max = self._golden_section_search(
            P_inlet, h_inlet, feed_temperature,
            P_array[max(0, idx_max-1)],
            P_array[min(n_points-1, idx_max+1)]
        )

        # Get exit conditions at critical pressure
        quality, rho_exit = self._get_two_phase_properties(
            P_critical, h_inlet, feed_temperature
        )

        # Calculate required area
        area = m_dot / (Cd * G_max)

        # Velocity
        velocity = G_max / rho_exit

        return {
            'area': area,
            'mass_flux': G_max,
            'velocity': velocity,
            'quality': quality,
            'P_critical': P_critical,
        }

    def _solve_hem_nonchoked(
        self,
        P_inlet: float,
        P_exit: float,
        h_inlet: float,
        m_dot: float,
        Cd: float,
        feed_temperature: float
    ) -> Dict[str, float]:
        """Solve for non-choked HEM flow (rare for N2O).

        Args:
            P_inlet: Inlet pressure (Pa)
            P_exit: Exit pressure (Pa)
            h_inlet: Inlet enthalpy (J/kg)
            m_dot: Mass flow rate (kg/s)
            Cd: Discharge coefficient
            feed_temperature: Feed temperature (K)

        Returns:
            dict with area, mass_flux, velocity, quality
        """
        # Calculate mass flux for expansion to exit pressure
        G = self._calculate_mass_flux_hem(P_inlet, P_exit, h_inlet, feed_temperature)

        # Get exit properties
        quality, rho_exit = self._get_two_phase_properties(P_exit, h_inlet, feed_temperature)

        # Area
        area = m_dot / (Cd * G)

        # Velocity
        velocity = G / rho_exit

        return {
            'area': area,
            'mass_flux': G,
            'velocity': velocity,
            'quality': quality,
        }

    def _calculate_mass_flux_hem(
        self,
        P1: float,
        P2: float,
        h_inlet: float,
        feed_temperature: float,
        n_steps: int = 20
    ) -> float:
        """Calculate HEM mass flux for expansion from P1 to P2.

        Integrates: G = sqrt(2 x integral(rho x dP))

        For two-phase flow, rho varies with pressure through the orifice.

        Args:
            P1: Inlet pressure (Pa)
            P2: Exit pressure (Pa)
            h_inlet: Inlet specific enthalpy (J/kg)
            feed_temperature: Feed temperature for property lookups (K)
            n_steps: Number of integration steps

        Returns:
            Mass flux G (kg/m^2-s)
        """
        if P2 >= P1:
            return 0.0

        # Pressure points for integration
        P_points = np.linspace(P1, P2, n_steps)

        # Calculate density at each pressure point
        # For HEM: constant enthalpy expansion (throttling)
        integral = 0.0

        for i in range(len(P_points) - 1):
            P_avg = (P_points[i] + P_points[i+1]) / 2
            dP = P_points[i] - P_points[i+1]  # Positive for expansion

            # Get two-phase density at this pressure
            try:
                quality, rho_avg = self._get_two_phase_properties(
                    P_avg, h_inlet, feed_temperature
                )
                integral += rho_avg * dP
            except:
                # If properties fail, use inlet liquid density as fallback
                inlet_sat = self.fluids.n2o_saturation_properties(feed_temperature)
                integral += inlet_sat['rho_liquid'] * dP

        # G = sqrt(2 x integral)
        G = np.sqrt(2 * integral) if integral > 0 else 0.0

        return G

    def _get_two_phase_properties(
        self,
        P: float,
        h: float,
        feed_temperature: float
    ) -> Tuple[float, float]:
        """Get two-phase properties at given pressure and enthalpy.

        For throttling: enthalpy is constant

        Args:
            P: Pressure (Pa)
            h: Specific enthalpy (J/kg)
            feed_temperature: Reference temperature for lookups (K)

        Returns:
            (quality, density) tuple
        """
        # Get saturation properties at this pressure
        T_sat = self._get_saturation_temp(P)
        sat_props = self.fluids.n2o_saturation_properties(T_sat)

        h_f = sat_props['h_liquid']
        h_g = sat_props['h_vapor']
        h_fg = sat_props['h_fg']
        rho_f = sat_props['rho_liquid']
        rho_g = sat_props['rho_vapor']

        # Calculate quality from enthalpy balance
        if abs(h_fg) < 1e-6:
            # At critical point
            quality = 0.5
            rho = (rho_f + rho_g) / 2
        elif h <= h_f:
            # Subcooled/compressed liquid
            quality = 0.0
            rho = rho_f
        elif h >= h_g:
            # Superheated vapor
            quality = 1.0
            rho = rho_g
        else:
            # Two-phase mixture
            quality = (h - h_f) / h_fg
            quality = np.clip(quality, 0.0, 1.0)

            # Homogeneous density: 1/rho = x/rho_g + (1-x)/rho_f
            rho = 1.0 / (quality/rho_g + (1-quality)/rho_f)

        return quality, rho

    def _golden_section_search(
        self,
        P_inlet: float,
        h_inlet: float,
        feed_temperature: float,
        P_low: float,
        P_high: float,
        tol: float = 1e-5
    ) -> Tuple[float, float]:
        """Find pressure that maximizes mass flux using golden section search.

        Args:
            P_inlet: Inlet pressure (Pa)
            h_inlet: Inlet enthalpy (J/kg)
            feed_temperature: Feed temperature (K)
            P_low: Lower bound on pressure (Pa)
            P_high: Upper bound on pressure (Pa)
            tol: Convergence tolerance

        Returns:
            (P_optimal, G_max) tuple
        """
        golden_ratio = (1 + np.sqrt(5)) / 2

        # Golden section points
        P1 = P_high - (P_high - P_low) / golden_ratio
        P2 = P_low + (P_high - P_low) / golden_ratio

        max_iter = 50
        for _ in range(max_iter):
            G1 = self._calculate_mass_flux_hem(P_inlet, P1, h_inlet, feed_temperature)
            G2 = self._calculate_mass_flux_hem(P_inlet, P2, h_inlet, feed_temperature)

            if G1 > G2:
                P_high = P2
                P2 = P1
                P1 = P_high - (P_high - P_low) / golden_ratio
            else:
                P_low = P1
                P1 = P2
                P2 = P_low + (P_high - P_low) / golden_ratio

            if abs(P_high - P_low) < tol:
                break

        P_optimal = (P_low + P_high) / 2
        G_max = self._calculate_mass_flux_hem(P_inlet, P_optimal, h_inlet, feed_temperature)

        return P_optimal, G_max

    def _get_saturation_temp(self, pressure: float) -> float:
        """Get saturation temperature for given pressure.

        Args:
            pressure: Pressure (Pa)

        Returns:
            Saturation temperature (K)
        """
        try:
            import CoolProp.CoolProp as CP
            T_sat = CP.PropsSI('T', 'P', pressure, 'Q', 0, 'N2O')
            return T_sat
        except:
            # Fallback: Antoine equation for N2O
            P_bar = pressure / 1e5
            if P_bar < 1 or P_bar > 100:
                raise ValueError(f"Pressure {P_bar:.1f} bar outside valid range")
            log_P = np.log10(P_bar)
            T_C = 1237.0 / (6.53 - log_P) - 260.0
            return T_C + 273.15

    def size_all_injectors(
        self,
        n2o_mass_flow: float,
        ethanol_mass_flow: float,
        n2o_feed_pressure: float,
        ethanol_feed_pressure: float,
        chamber_pressure: float,
        n2o_orifice_count: int,
        ethanol_orifice_count: int,
        discharge_coefficient: float,
        n2o_feed_temperature: float = 293.15,
        ethanol_feed_temperature: float = 298.15
    ) -> Dict[str, Dict[str, float]]:
        """Size both N2O and ethanol injectors."""

        n2o_results = self.size_n2o_orifice_hem(
            mass_flow=n2o_mass_flow,
            feed_pressure=n2o_feed_pressure,
            chamber_pressure=chamber_pressure,
            orifice_count=n2o_orifice_count,
            discharge_coefficient=discharge_coefficient,
            feed_temperature=n2o_feed_temperature
        )

        ethanol_results = self.size_ethanol_orifice(
            mass_flow=ethanol_mass_flow,
            feed_pressure=ethanol_feed_pressure,
            chamber_pressure=chamber_pressure,
            orifice_count=ethanol_orifice_count,
            discharge_coefficient=discharge_coefficient,
            feed_temperature=ethanol_feed_temperature
        )

        return {
            'n2o': n2o_results,
            'ethanol': ethanol_results,
        }
