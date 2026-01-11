"""
Nozzle sizing for torch igniter.

Handles throat area calculation and exit geometry
for both conical and bell nozzles.
"""

import numpy as np
from typing import Dict

from .utils import circle_area, diameter_from_area, conical_nozzle_length


class NozzleDesigner:
    """Nozzle sizing and geometry calculations."""
    
    def __init__(self, expansion_ratio: float = 3.0,
                 nozzle_type: str = "conical",
                 conical_half_angle: float = 15.0):
        """Initialize nozzle designer.
        
        Args:
            expansion_ratio: Area ratio Ae/At
            nozzle_type: "conical" or "bell"
            conical_half_angle: Half angle for conical nozzle (degrees)
        """
        self.expansion_ratio = expansion_ratio
        self.nozzle_type = nozzle_type
        self.conical_half_angle = conical_half_angle
    
    def calculate_throat_area(self, mass_flow: float, chamber_pressure: float,
                              c_star: float) -> float:
        """Calculate required throat area from mass flow.
        
        Uses fundamental relation: m_dot = P_c * A_t / c*
        
        Args:
            mass_flow: Total mass flow rate (kg/s)
            chamber_pressure: Chamber stagnation pressure (Pa)
            c_star: Characteristic velocity (m/s)
            
        Returns:
            Throat area (m^2)
        """
        # From m_dot = P_c * A_t / c*
        # A_t = m_dot * c* / P_c
        throat_area = mass_flow * c_star / chamber_pressure
        return throat_area
    
    def calculate_exit_area(self, throat_area: float) -> float:
        """Calculate exit area from expansion ratio.
        
        Args:
            throat_area: Throat area (m^2)
            
        Returns:
            Exit area (m^2)
        """
        return throat_area * self.expansion_ratio
    
    def calculate_nozzle_geometry(self, throat_area: float) -> Dict[str, float]:
        """Calculate complete nozzle geometry.
        
        Args:
            throat_area: Throat area (m^2)
            
        Returns:
            Dictionary with:
            - throat_diameter: Throat diameter (m)
            - throat_area: Throat area (m^2)
            - exit_diameter: Exit diameter (m)
            - exit_area: Exit area (m^2)
            - nozzle_length: Divergent section length (m)
            - expansion_ratio: Ae/At
        """
        throat_diameter = diameter_from_area(throat_area)
        exit_area = self.calculate_exit_area(throat_area)
        exit_diameter = diameter_from_area(exit_area)
        
        # Calculate nozzle length based on type
        if self.nozzle_type == "conical":
            nozzle_length = conical_nozzle_length(
                throat_diameter,
                exit_diameter,
                self.conical_half_angle
            )
        elif self.nozzle_type == "bell":
            # For bell nozzle, use 80% of equivalent conical length
            # (this is a rough approximation)
            conical_length = conical_nozzle_length(
                throat_diameter,
                exit_diameter,
                self.conical_half_angle
            )
            nozzle_length = 0.8 * conical_length
        else:
            raise ValueError(f"Unknown nozzle type: {self.nozzle_type}")
        
        return {
            'throat_diameter': throat_diameter,
            'throat_area': throat_area,
            'exit_diameter': exit_diameter,
            'exit_area': exit_area,
            'nozzle_length': nozzle_length,
            'expansion_ratio': self.expansion_ratio,
        }
    
    def calculate_exit_conditions(self, chamber_pressure: float,
                                  gamma: float,
                                  gamma_exit: float = None) -> Dict[str, float]:
        """Calculate exit plane conditions.
        
        Args:
            chamber_pressure: Chamber pressure (Pa)
            gamma: Ratio of specific heats (at throat)
            gamma_exit: Ratio of specific heats at exit (if None, use gamma)
            
        Returns:
            Dictionary with:
            - exit_pressure: Static pressure at exit (Pa)
            - exit_mach: Mach number at exit
        """
        if gamma_exit is None:
            gamma_exit = gamma
        
        # For isentropic expansion, use area-Mach relation
        # A/A* = (1/M) * [(2/(gamma+1)) * (1 + (gamma-1)/2 * M^2)]^((gamma+1)/(2*(gamma-1)))
        # This needs to be solved iteratively for Mach number
        
        # Approximate solution using correlation for supersonic flow
        # M_exit ≈ sqrt((2/(gamma-1)) * ((A_exit/A_throat)^((gamma-1)/gamma) - 1))
        
        # More accurate: solve iteratively
        exit_mach = self._solve_exit_mach(self.expansion_ratio, gamma_exit)
        
        # Calculate exit pressure from isentropic relation
        # P_exit / P_chamber = (1 + (gamma-1)/2 * M^2)^(-gamma/(gamma-1))
        pressure_ratio = (1 + (gamma_exit - 1) / 2 * exit_mach**2) ** (
            -gamma_exit / (gamma_exit - 1)
        )
        exit_pressure = chamber_pressure * pressure_ratio
        
        return {
            'exit_pressure': exit_pressure,
            'exit_mach': exit_mach,
            'pressure_ratio': pressure_ratio,
        }
    
    def _solve_exit_mach(self, area_ratio: float, gamma: float,
                         tol: float = 1e-6, max_iter: int = 100) -> float:
        """Solve for exit Mach number from area ratio.
        
        Uses Newton-Raphson iteration on area-Mach relation.
        
        Args:
            area_ratio: A_exit / A_throat
            gamma: Ratio of specific heats
            tol: Convergence tolerance
            max_iter: Maximum iterations
            
        Returns:
            Exit Mach number
        """
        # Initial guess (supersonic branch)
        M = np.sqrt(2 / (gamma - 1) * (area_ratio**(2*(gamma-1)/gamma) - 1))
        M = max(M, 1.5)  # Ensure we're on supersonic branch
        
        for _ in range(max_iter):
            # Area-Mach function
            A_M = (1/M) * ((2/(gamma+1)) * (1 + (gamma-1)/2 * M**2)) ** (
                (gamma+1) / (2*(gamma-1))
            )
            
            # Residual
            f = A_M - area_ratio
            
            if abs(f) < tol:
                break
            
            # Derivative
            term1 = -1/M**2
            term2 = (1/(gamma+1)) * ((2/(gamma+1)) * (1 + (gamma-1)/2 * M**2)) ** (
                (gamma+1)/(2*(gamma-1)) - 1
            )
            term3 = (2/(gamma+1)) * (gamma-1) * M
            
            df_dM = term1 * ((2/(gamma+1)) * (1 + (gamma-1)/2 * M**2)) ** (
                (gamma+1)/(2*(gamma-1))
            ) + (1/M) * term2 * term3
            
            # Newton-Raphson update
            M = M - f / df_dM
        
        return M
    
    def calculate_thrust(self, mass_flow: float, exit_velocity: float,
                        exit_pressure: float, exit_area: float,
                        ambient_pressure: float) -> float:
        """Calculate thrust from exit conditions.
        
        F = m_dot * v_exit + (P_exit - P_ambient) * A_exit
        
        Args:
            mass_flow: Mass flow rate (kg/s)
            exit_velocity: Exit velocity (m/s)
            exit_pressure: Exit static pressure (Pa)
            exit_area: Exit area (m^2)
            ambient_pressure: Ambient pressure (Pa)
            
        Returns:
            Thrust (N)
        """
        momentum_thrust = mass_flow * exit_velocity
        pressure_thrust = (exit_pressure - ambient_pressure) * exit_area
        total_thrust = momentum_thrust + pressure_thrust
        
        return total_thrust


def size_nozzle_from_mass_flow(
    mass_flow: float,
    chamber_pressure: float,
    c_star: float,
    expansion_ratio: float = 3.0,
    nozzle_type: str = "conical",
    conical_half_angle: float = 15.0
) -> Dict[str, float]:
    """Size complete nozzle from operating conditions.
    
    Convenience function combining throat and exit sizing.
    
    Args:
        mass_flow: Total mass flow rate (kg/s)
        chamber_pressure: Chamber pressure (Pa)
        c_star: Characteristic velocity (m/s)
        expansion_ratio: Area ratio
        nozzle_type: "conical" or "bell"
        conical_half_angle: Cone half angle (degrees)
        
    Returns:
        Dictionary with complete nozzle geometry
    """
    designer = NozzleDesigner(
        expansion_ratio=expansion_ratio,
        nozzle_type=nozzle_type,
        conical_half_angle=conical_half_angle
    )
    
    throat_area = designer.calculate_throat_area(
        mass_flow, chamber_pressure, c_star
    )
    
    geometry = designer.calculate_nozzle_geometry(throat_area)
    
    return geometry
