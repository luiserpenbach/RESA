"""
Torch Igniter Sizing Tool

A preliminary design tool for bipropellant torch igniters using
Ethanol and Nitrous Oxide propellants.

Key features:
- NASA CEA integration for combustion calculations
- CoolProp for fluid properties
- L* method chamber sizing
- Simple injector orifice sizing
- Performance analysis and operating envelopes
"""

__version__ = "0.1.0"

from .config import IgniterConfig, IgniterResults
from .cea_interface import CEACalculator, estimate_heat_power
from .fluids import FluidProperties, get_ethanol_density, get_n2o_density
from .chamber import ChamberDesigner, size_chamber_from_mass_flow
from .nozzle import NozzleDesigner, size_nozzle_from_mass_flow
from .performance import PerformanceCalculator, calculate_all_performance_metrics
from .injector import InjectorDesigner

__all__ = [
    'IgniterConfig',
    'IgniterResults',
    'IgniterDesigner',
    'CEACalculator',
    'FluidProperties',
    'ChamberDesigner',
    'NozzleDesigner',
    'PerformanceCalculator',
    'InjectorDesigner',
]


class IgniterDesigner:
    """Main igniter design interface.

    This class orchestrates the complete igniter design process,
    from configuration to final results.

    Example:
        >>> config = IgniterConfig(
        ...     chamber_pressure=20e5,
        ...     mixture_ratio=2.0,
        ...     total_mass_flow=0.050,
        ...     ethanol_feed_pressure=25e5,
        ...     n2o_feed_pressure=30e5,
        ...     ethanol_feed_temperature=298.15,
        ...     n2o_feed_temperature=298.15
        ... )
        >>> designer = IgniterDesigner()
        >>> results = designer.design(config)
        >>> print(results.summary())
    """

    def __init__(self):
        """Initialize designer with required calculators."""
        self.cea = CEACalculator()
        self.fluids = FluidProperties()
        self.injector = InjectorDesigner()

    def design(self, config: IgniterConfig) -> IgniterResults:
        """Design complete igniter from configuration.

        Args:
            config: Igniter configuration

        Returns:
            Complete design results
        """
        # Step 1: Get combustion properties from CEA
        cea_props = self.cea.get_combustion_properties(
            mixture_ratio=config.mixture_ratio,
            chamber_pressure_pa=config.chamber_pressure,
            expansion_ratio=config.expansion_ratio,
            frozen=False
        )

        # Step 2: Calculate heat power output using LHV method
        heat_power = estimate_heat_power(
            mass_flow_total=config.total_mass_flow,
            mixture_ratio=config.mixture_ratio
        )

        # Step 3: Size nozzle (throat and exit)
        nozzle_geom = size_nozzle_from_mass_flow(
            mass_flow=config.total_mass_flow,
            chamber_pressure=config.chamber_pressure,
            c_star=cea_props['c_star'],
            expansion_ratio=config.expansion_ratio,
            nozzle_type=config.nozzle_type,
            conical_half_angle=config.conical_half_angle
        )

        # Step 4: Size chamber using L* method
        chamber_geom = size_chamber_from_mass_flow(
            mass_flow=config.total_mass_flow,
            chamber_pressure=config.chamber_pressure,
            c_star=cea_props['c_star'],
            l_star=config.l_star,
            ld_ratio=3.0  # Fixed L/D ratio for now
        )

        # Step 5: Size injector orifices using HEM for N2O
        injector_results = self.injector.size_all_injectors(
            n2o_mass_flow=config.oxidizer_mass_flow,
            ethanol_mass_flow=config.fuel_mass_flow,
            n2o_feed_pressure=config.n2o_feed_pressure,
            ethanol_feed_pressure=config.ethanol_feed_pressure,
            chamber_pressure=config.chamber_pressure,
            n2o_orifice_count=config.n2o_orifice_count,
            ethanol_orifice_count=config.ethanol_orifice_count,
            discharge_coefficient=config.discharge_coefficient,
            n2o_feed_temperature=config.n2o_feed_temperature,
            ethanol_feed_temperature=config.ethanol_feed_temperature
        )

        # Step 6: Calculate performance metrics
        perf_metrics = calculate_all_performance_metrics(
            mass_flow=config.total_mass_flow,
            chamber_pressure=config.chamber_pressure,
            throat_area=nozzle_geom['throat_area'],
            chamber_volume=chamber_geom['volume'],
            chamber_diameter=chamber_geom['diameter'],
            chamber_length=chamber_geom['length'],
            c_star_theoretical=cea_props['c_star'],
            isp_theoretical=cea_props['isp'],
            mixture_ratio=config.mixture_ratio,
            ambient_pressure=config.ambient_pressure
        )

        # Step 7: Assemble results
        results = IgniterResults(
            # Combustion
            flame_temperature=cea_props['T_chamber'],
            c_star=cea_props['c_star'],
            gamma=cea_props['gamma'],
            molecular_weight=cea_props['MW'],
            heat_power_output=heat_power,

            # Geometry
            chamber_diameter=chamber_geom['diameter'],
            chamber_length=chamber_geom['length'],
            chamber_volume=chamber_geom['volume'],
            throat_diameter=nozzle_geom['throat_diameter'],
            throat_area=nozzle_geom['throat_area'],
            exit_diameter=nozzle_geom['exit_diameter'],
            exit_area=nozzle_geom['exit_area'],
            nozzle_length=nozzle_geom['nozzle_length'],

            # Injector - now using HEM for N2O
            n2o_orifice_diameter=injector_results['n2o']['orifice_diameter'],
            ethanol_orifice_diameter=injector_results['ethanol']['orifice_diameter'],
            n2o_injection_velocity=injector_results['n2o']['injection_velocity'],
            ethanol_injection_velocity=injector_results['ethanol']['injection_velocity'],
            n2o_pressure_drop=injector_results['n2o']['pressure_drop'],
            ethanol_pressure_drop=injector_results['ethanol']['pressure_drop'],

            # Performance
            isp_theoretical=cea_props['isp'],
            thrust=perf_metrics['thrust'],

            # Reference values
            oxidizer_mass_flow=config.oxidizer_mass_flow,
            fuel_mass_flow=config.fuel_mass_flow,
            total_mass_flow=config.total_mass_flow,
            mixture_ratio=config.mixture_ratio,
            chamber_pressure=config.chamber_pressure,

            # Optional/calculated
            c_star_efficiency=perf_metrics['c_star_efficiency'],
        )

        return results