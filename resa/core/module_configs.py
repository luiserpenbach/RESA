"""
Module-specific configuration dataclasses for RESA.

Keeps EngineConfig focused on core engine parameters while providing
dedicated configs for downstream analysis modules.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CoolingChannelConfig:
    """Extended cooling channel design parameters."""

    channel_type: str = "rectangular"  # 'rectangular', 'trapezoidal'
    num_channels_override: Optional[int] = None

    # Height profile
    height_profile: str = "constant"  # 'constant', 'tapered', 'custom'
    height_throat_m: float = 0.75e-3
    height_chamber_m: float = 1.5e-3
    height_exit_m: float = 1.0e-3

    # Trapezoidal channel parameters
    taper_angle_deg: float = 10.0  # sidewall taper angle for trapezoidal channels

    # Bifurcation
    bifurcation_enabled: bool = False
    bifurcation_station_x: float = 0.0  # axial position where channels split [m]

    # Constraints
    aspect_ratio_limit: float = 10.0

    # Optimization
    optimization_target: str = "none"  # 'none', 'min_dp', 'min_wall_temp', 'max_margin'

    # Wall geometry overrides (None = use EngineConfig value)
    wall_thickness_mm: Optional[float] = None     # hot wall thickness [mm]
    rib_width_throat_mm: Optional[float] = None   # rib width at throat [mm]

    # Axial margins for CAD import
    start_margin_mm: float = 0.0  # channels begin this far from injector face [mm]
    end_margin_mm: float = 0.0    # channels end this far before nozzle exit [mm]

    # Surface roughness override (None = use EngineConfig value)
    roughness_microns: Optional[float] = None

    # Helix/spiral angle
    helix_angle_deg: float = 0.0

    # Coolant inlet overrides (None = use EngineConfig value)
    coolant_p_in_bar: Optional[float] = None
    coolant_t_in_k: Optional[float] = None
    coolant_mass_fraction: Optional[float] = None


@dataclass
class WallThicknessConfig:
    """Configuration for wall thickness / structural analysis."""

    material_name: str = "inconel718"
    safety_factor_pressure: float = 2.0
    safety_factor_thermal: float = 1.5
    design_life_cycles: int = 100
    include_fatigue: bool = False


@dataclass
class PerformanceMapConfig:
    """Configuration for off-design performance map generation."""

    # Altitude sweep
    altitude_range_m: Tuple[float, float] = (0, 100_000)
    altitude_points: int = 50

    # Throttle sweep
    throttle_range: Tuple[float, float] = (0.3, 1.0)
    throttle_points: int = 15
    throttle_mode: str = "both"  # 'ox_only', 'fuel_only', 'both'

    # MR sweep
    mr_sweep_range: Tuple[float, float] = (2.0, 8.0)
    mr_sweep_points: int = 20


@dataclass
class FeedSystemConfig:
    """Configuration for feed system analysis."""

    feed_type: str = "pressure-fed"  # 'pressure-fed', 'pump-fed'
    cycle_type: str = "none"  # 'none', 'gas-generator', 'expander', 'staged-combustion'

    # Oxidizer line
    ox_line_length_m: float = 2.0
    ox_line_diameter_m: float = 0.012
    ox_line_roughness_m: float = 15e-6
    ox_k_fittings: float = 5.0

    # Fuel line
    fuel_line_length_m: float = 1.5
    fuel_line_diameter_m: float = 0.010
    fuel_line_roughness_m: float = 15e-6
    fuel_k_fittings: float = 4.0

    # Tank / pressurization
    tank_pressure_bar: float = 0.0  # 0 = auto-calculate
    ullage_pressure_bar: float = 5.0

    # Pump parameters (used when feed_type='pump-fed')
    pump_efficiency: float = 0.65
    turbine_efficiency: float = 0.60

    # Gas generator parameters (used when cycle_type='gas-generator')
    gg_temperature_k: float = 900.0
    gg_mr: float = 0.3  # fuel-rich GG mixture ratio

    # Suction parameters
    suction_head_m: float = 1.0
