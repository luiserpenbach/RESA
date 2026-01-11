"""
Analysis tools for torch igniter performance mapping.

Generates operating envelopes by sweeping design parameters
and creating performance maps.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .config import IgniterConfig
from .cea_interface import CEACalculator


@dataclass
class OperatingPoint:
    """Single point in operating envelope."""
    mixture_ratio: float
    chamber_pressure: float
    mass_flow: float
    c_star: float
    isp: float
    flame_temperature: float
    gamma: float
    thrust: float
    heat_power: float
    n2o_mass_flow: float
    ethanol_mass_flow: float


class EnvelopeGenerator:
    """Generate operating envelopes for torch igniter."""
    
    def __init__(self):
        """Initialize with CEA calculator."""
        self.cea = CEACalculator()
    
    def generate_mixture_ratio_sweep(
        self,
        base_config: IgniterConfig,
        mr_range: Tuple[float, float],
        n_points: int = 15
    ) -> pd.DataFrame:
        """Sweep mixture ratio at constant chamber pressure.
        
        Args:
            base_config: Baseline configuration
            mr_range: (min, max) mixture ratio
            n_points: Number of points
            
        Returns:
            DataFrame with performance data
        """
        mr_values = np.linspace(mr_range[0], mr_range[1], n_points)
        
        results = []
        for mr in mr_values:
            # Get combustion properties
            cea_props = self.cea.get_combustion_properties(
                mixture_ratio=mr,
                chamber_pressure_pa=base_config.chamber_pressure,
                expansion_ratio=base_config.expansion_ratio
            )
            
            # Calculate mass flows
            m_ox = base_config.total_mass_flow * mr / (1 + mr)
            m_fuel = base_config.total_mass_flow / (1 + mr)
            
            # Heat power
            from .cea_interface import estimate_heat_power
            heat_power = estimate_heat_power(
                base_config.total_mass_flow, mr
            )
            
            # Thrust
            thrust = base_config.total_mass_flow * cea_props['isp'] * 9.80665
            
            results.append({
                'mixture_ratio': mr,
                'chamber_pressure': base_config.chamber_pressure / 1e5,  # bar
                'total_mass_flow': base_config.total_mass_flow * 1000,  # g/s
                'c_star': cea_props['c_star'],
                'isp': cea_props['isp'],
                'flame_temperature': cea_props['T_chamber'],
                'gamma': cea_props['gamma'],
                'thrust': thrust,
                'heat_power': heat_power / 1000,  # kW
                'n2o_mass_flow': m_ox * 1000,  # g/s
                'ethanol_mass_flow': m_fuel * 1000,  # g/s
            })
        
        return pd.DataFrame(results)
    
    def generate_pressure_sweep(
        self,
        base_config: IgniterConfig,
        pressure_range: Tuple[float, float],
        n_points: int = 15
    ) -> pd.DataFrame:
        """Sweep chamber pressure at constant mixture ratio.
        
        Args:
            base_config: Baseline configuration
            pressure_range: (min, max) pressure in Pa
            n_points: Number of points
            
        Returns:
            DataFrame with performance data
        """
        p_values = np.linspace(pressure_range[0], pressure_range[1], n_points)
        
        results = []
        for p in p_values:
            # Get combustion properties
            cea_props = self.cea.get_combustion_properties(
                mixture_ratio=base_config.mixture_ratio,
                chamber_pressure_pa=p,
                expansion_ratio=base_config.expansion_ratio
            )
            
            # Heat power
            from .cea_interface import estimate_heat_power
            heat_power = estimate_heat_power(
                base_config.total_mass_flow,
                base_config.mixture_ratio
            )
            
            # Thrust
            thrust = base_config.total_mass_flow * cea_props['isp'] * 9.80665
            
            results.append({
                'mixture_ratio': base_config.mixture_ratio,
                'chamber_pressure': p / 1e5,  # bar
                'total_mass_flow': base_config.total_mass_flow * 1000,  # g/s
                'c_star': cea_props['c_star'],
                'isp': cea_props['isp'],
                'flame_temperature': cea_props['T_chamber'],
                'gamma': cea_props['gamma'],
                'thrust': thrust,
                'heat_power': heat_power / 1000,  # kW
                'n2o_mass_flow': base_config.oxidizer_mass_flow * 1000,
                'ethanol_mass_flow': base_config.fuel_mass_flow * 1000,
            })
        
        return pd.DataFrame(results)
    
    def generate_2d_envelope(
        self,
        base_config: IgniterConfig,
        mr_range: Tuple[float, float],
        pressure_range: Tuple[float, float],
        n_mr: int = 10,
        n_p: int = 10
    ) -> pd.DataFrame:
        """Generate 2D operating envelope (mixture ratio vs. pressure).
        
        Args:
            base_config: Baseline configuration
            mr_range: (min, max) mixture ratio
            pressure_range: (min, max) pressure in Pa
            n_mr: Number of mixture ratio points
            n_p: Number of pressure points
            
        Returns:
            DataFrame with performance grid
        """
        mr_values = np.linspace(mr_range[0], mr_range[1], n_mr)
        p_values = np.linspace(pressure_range[0], pressure_range[1], n_p)
        
        results = []
        for mr in mr_values:
            for p in p_values:
                try:
                    # Get combustion properties
                    cea_props = self.cea.get_combustion_properties(
                        mixture_ratio=mr,
                        chamber_pressure_pa=p,
                        expansion_ratio=base_config.expansion_ratio
                    )
                    
                    # Mass flows
                    m_ox = base_config.total_mass_flow * mr / (1 + mr)
                    m_fuel = base_config.total_mass_flow / (1 + mr)
                    
                    # Heat power
                    from .cea_interface import estimate_heat_power
                    heat_power = estimate_heat_power(
                        base_config.total_mass_flow, mr
                    )
                    
                    # Thrust
                    thrust = base_config.total_mass_flow * cea_props['isp'] * 9.80665
                    
                    results.append({
                        'mixture_ratio': mr,
                        'chamber_pressure': p / 1e5,  # bar
                        'total_mass_flow': base_config.total_mass_flow * 1000,  # g/s
                        'c_star': cea_props['c_star'],
                        'isp': cea_props['isp'],
                        'flame_temperature': cea_props['T_chamber'],
                        'gamma': cea_props['gamma'],
                        'thrust': thrust,
                        'heat_power': heat_power / 1000,  # kW
                        'n2o_mass_flow': m_ox * 1000,
                        'ethanol_mass_flow': m_fuel * 1000,
                    })
                    
                except Exception as e:
                    # Skip points that fail (e.g., out of CEA range)
                    print(f"Warning: Failed for MR={mr:.2f}, P={p/1e5:.1f} bar: {e}")
                    continue
        
        return pd.DataFrame(results)
    
    def generate_mass_flow_sweep(
        self,
        base_config: IgniterConfig,
        mass_flow_range: Tuple[float, float],
        n_points: int = 15
    ) -> pd.DataFrame:
        """Sweep total mass flow at constant mixture ratio and pressure.
        
        Args:
            base_config: Baseline configuration
            mass_flow_range: (min, max) mass flow in kg/s
            n_points: Number of points
            
        Returns:
            DataFrame with performance data
        """
        m_values = np.linspace(mass_flow_range[0], mass_flow_range[1], n_points)
        
        # Get combustion properties (independent of mass flow)
        cea_props = self.cea.get_combustion_properties(
            mixture_ratio=base_config.mixture_ratio,
            chamber_pressure_pa=base_config.chamber_pressure,
            expansion_ratio=base_config.expansion_ratio
        )
        
        results = []
        for m_total in m_values:
            # Mass flows
            m_ox = m_total * base_config.mixture_ratio / (1 + base_config.mixture_ratio)
            m_fuel = m_total / (1 + base_config.mixture_ratio)
            
            # Heat power
            from .cea_interface import estimate_heat_power
            heat_power = estimate_heat_power(m_total, base_config.mixture_ratio)
            
            # Thrust
            thrust = m_total * cea_props['isp'] * 9.80665
            
            results.append({
                'mixture_ratio': base_config.mixture_ratio,
                'chamber_pressure': base_config.chamber_pressure / 1e5,  # bar
                'total_mass_flow': m_total * 1000,  # g/s
                'c_star': cea_props['c_star'],
                'isp': cea_props['isp'],
                'flame_temperature': cea_props['T_chamber'],
                'gamma': cea_props['gamma'],
                'thrust': thrust,
                'heat_power': heat_power / 1000,  # kW
                'n2o_mass_flow': m_ox * 1000,
                'ethanol_mass_flow': m_fuel * 1000,
            })
        
        return pd.DataFrame(results)


def find_optimal_mixture_ratio(
    base_config: IgniterConfig,
    mr_range: Tuple[float, float] = (1.5, 3.0),
    objective: str = 'isp'
) -> Tuple[float, pd.DataFrame]:
    """Find optimal mixture ratio for given objective.
    
    Args:
        base_config: Baseline configuration
        mr_range: Search range for mixture ratio
        objective: 'isp', 'c_star', or 'heat_power'
        
    Returns:
        (optimal_mr, sweep_data) tuple
    """
    generator = EnvelopeGenerator()
    sweep = generator.generate_mixture_ratio_sweep(base_config, mr_range, n_points=30)
    
    if objective not in sweep.columns:
        raise ValueError(f"Unknown objective: {objective}")
    
    # Find maximum
    idx_max = sweep[objective].idxmax()
    optimal_mr = sweep.loc[idx_max, 'mixture_ratio']
    
    return optimal_mr, sweep
