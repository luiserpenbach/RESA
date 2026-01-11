"""
Chamber sizing using L* (characteristic length) methodology.

The L* method is the industry-standard approach for ensuring
adequate combustion chamber volume and residence time.
"""

import numpy as np
from typing import Dict

from .nozzle import circle_area, diameter_from_area


class ChamberDesigner:
    """Chamber sizing using L* method.

    The chamber volume is sized based on characteristic length L*,
    which is defined as: L* = V_chamber / A_throat

    Typical L* values for igniters range from 0.5 to 1.5 meters.
    """

    def __init__(self, l_star: float = 1.0, ld_ratio: float = 3.0):
        """Initialize chamber designer.

        Args:
            l_star: Characteristic length (m), typical 0.5-1.5 for igniters
            ld_ratio: Chamber L/D ratio, typical 2-4
        """
        self.l_star = l_star
        self.ld_ratio = ld_ratio

    def size_chamber(self, throat_area: float) -> Dict[str, float]:
        """Calculate chamber geometry from throat area and L*.

        Args:
            throat_area: Nozzle throat area (m^2)

        Returns:
            Dictionary with:
            - volume: Chamber volume (m^3)
            - diameter: Chamber diameter (m)
            - length: Chamber length (m)
            - area: Chamber cross-sectional area (m^2)
        """
        # Calculate required chamber volume
        volume = self.l_star * throat_area

        # Calculate chamber dimensions from L/D ratio
        # V = pi * D^2 / 4 * L
        # L = L/D * D
        # V = pi * D^2 / 4 * L/D * D = pi/4 * (L/D) * D^3
        # D^3 = V * 4 / (pi * L/D)
        diameter = (volume * 4 / (np.pi * self.ld_ratio)) ** (1/3)
        length = self.ld_ratio * diameter
        area = circle_area(diameter)

        return {
            'volume': volume,
            'diameter': diameter,
            'length': length,
            'area': area,
        }

    def calculate_stay_time(self, throat_area: float, c_star: float,
                            chamber_pressure: float) -> float:
        """Calculate characteristic residence time in chamber.

        Args:
            throat_area: Throat area (m^2)
            c_star: Characteristic velocity (m/s)
            chamber_pressure: Chamber pressure (Pa)

        Returns:
            Stay time tau (s)

        Note:
            tau = V_chamber / (A_throat * c_star)
                = L* / c_star
        """
        return self.l_star / c_star

    def calculate_contraction_ratio(self, chamber_area: float,
                                    throat_area: float) -> float:
        """Calculate chamber contraction ratio.

        Args:
            chamber_area: Chamber cross-sectional area (m^2)
            throat_area: Throat area (m^2)

        Returns:
            Contraction ratio (dimensionless)
        """
        return chamber_area / throat_area


def size_chamber_from_mass_flow(
    mass_flow: float,
    chamber_pressure: float,
    c_star: float,
    l_star: float = 1.0,
    ld_ratio: float = 3.0
) -> Dict[str, float]:
    """Size chamber directly from mass flow and operating conditions.

    This is a convenience function that combines throat sizing
    with chamber sizing in one step.

    Args:
        mass_flow: Total mass flow rate (kg/s)
        chamber_pressure: Chamber pressure (Pa)
        c_star: Characteristic velocity (m/s)
        l_star: Characteristic length (m)
        ld_ratio: Chamber L/D ratio

    Returns:
        Dictionary with chamber geometry and throat area
    """
    # Calculate throat area from mass flow
    # m_dot = P_c * A_t / c*
    throat_area = mass_flow * c_star / chamber_pressure

    # Size chamber
    designer = ChamberDesigner(l_star=l_star, ld_ratio=ld_ratio)
    chamber_geom = designer.size_chamber(throat_area)

    # Add throat information
    chamber_geom['throat_area'] = throat_area
    chamber_geom['throat_diameter'] = diameter_from_area(throat_area)

    return chamber_geom


def validate_chamber_design(
    chamber_volume: float,
    throat_area: float,
    c_star: float,
    expected_l_star_range: tuple = (0.3, 2.0),
    expected_stay_time_range: tuple = (0.5e-3, 5e-3)
) -> Dict[str, bool]:
    """Validate chamber design against typical ranges.

    Args:
        chamber_volume: Chamber volume (m^3)
        throat_area: Throat area (m^2)
        c_star: Characteristic velocity (m/s)
        expected_l_star_range: Expected (min, max) for L* (m)
        expected_stay_time_range: Expected (min, max) for stay time (s)

    Returns:
        Dictionary with validation results
    """
    # Calculate actual L*
    l_star_actual = chamber_volume / throat_area

    # Calculate stay time
    stay_time = l_star_actual / c_star

    # Check ranges
    l_star_ok = expected_l_star_range[0] <= l_star_actual <= expected_l_star_range[1]
    stay_time_ok = (expected_stay_time_range[0] <= stay_time
                    <= expected_stay_time_range[1])

    return {
        'l_star_in_range': l_star_ok,
        'stay_time_in_range': stay_time_ok,
        'l_star_actual': l_star_actual,
        'stay_time_actual': stay_time,
        'all_ok': l_star_ok and stay_time_ok,
    }
