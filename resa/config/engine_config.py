"""
Engine configuration dataclass.

This module provides the EngineConfig dataclass which holds all parameters
needed to define a liquid rocket engine design.
"""

from dataclasses import dataclass, field
from typing import Optional
import os
import yaml


@dataclass
class EngineConfig:
    """
    Master configuration for a liquid rocket engine design.

    This dataclass holds all parameters needed for engine sizing,
    combustion analysis, geometry generation, and cooling analysis.

    Attributes:
        engine_name: Identifier for this engine design
        fuel: Fuel name (e.g., "Ethanol90", "RP1", "CH4")
        oxidizer: Oxidizer name (e.g., "N2O", "LOX")
        thrust_n: Target thrust in Newtons
        pc_bar: Chamber pressure in bar
        mr: Mixture ratio (O/F)

    Example:
        config = EngineConfig(
            engine_name="Phoenix-1",
            fuel="Ethanol90",
            oxidizer="N2O",
            thrust_n=2200.0,
            pc_bar=25.0,
            mr=4.0
        )
    """

    # --- Required Parameters ---
    engine_name: str
    fuel: str
    oxidizer: str
    thrust_n: float
    pc_bar: float
    mr: float

    # --- Nozzle / Chamber ---
    throat_diameter: float = 0.0    # [m] Optional - if 0, calculated from thrust
    expansion_ratio: float = 0.0    # If 0, calculated for optimal at p_exit_bar
    p_exit_bar: float = 1.013       # Design exit pressure [bar]
    L_star: float = 1100.0          # Characteristic length [mm]
    contraction_ratio: float = 10.0
    eff_combustion: float = 0.95    # Combustion efficiency
    theta_convergent: float = 30.0  # Convergent half-angle [deg]
    bell_fraction: float = 0.8      # Bell nozzle fraction (0.6-1.0)

    # --- Cooling System ---
    coolant_name: str = "REFPROP::NitrousOxide"
    cooling_mode: str = 'counter-flow'  # 'counter-flow' or 'co-flow'

    # Channel dimensions at throat [m]
    channel_width_throat: float = 1.0e-3
    channel_height: float = 0.75e-3
    rib_width_throat: float = 0.6e-3

    # Wall properties
    wall_thickness: float = 0.5e-3      # [m]
    wall_roughness: float = 20e-6       # [m]
    wall_conductivity: float = 15.0     # [W/(m·K)]

    # Coolant inlet conditions
    coolant_p_in_bar: float = 98.0      # Inlet pressure [bar]
    coolant_t_in_k: float = 290.0       # Inlet temperature [K]
    coolant_mass_fraction: float = 1.0  # Fraction of propellant flow used

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        if self.thrust_n <= 0 and self.throat_diameter <= 0:
            raise ValueError("Either thrust_n or throat_diameter must be positive")

        if self.pc_bar <= 0:
            raise ValueError("Chamber pressure must be positive")

        if self.mr <= 0:
            raise ValueError("Mixture ratio must be positive")

        if not 0 < self.eff_combustion <= 1.0:
            raise ValueError("Combustion efficiency must be between 0 and 1")

        if not 0.6 <= self.bell_fraction <= 1.0:
            raise ValueError("Bell fraction must be between 0.6 and 1.0")

        if self.cooling_mode not in ('counter-flow', 'co-flow'):
            raise ValueError("Cooling mode must be 'counter-flow' or 'co-flow'")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'EngineConfig':
        """
        Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            EngineConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required parameters are missing
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Helper for nested access
        def get(section: str, key: str, default=None):
            return data.get(section, {}).get(key, default)

        # Extract sections
        meta = data.get('meta', {})
        prop = data.get('propulsion', {})
        nozz = data.get('nozzle', {})
        cool = data.get('cooling', {})
        geo = cool.get('geometry', {})
        inlet = cool.get('inlet', {})

        return cls(
            # Meta
            engine_name=meta.get('engine_name', 'Unnamed Engine'),

            # Propulsion
            fuel=prop.get('fuel', 'Ethanol'),
            oxidizer=prop.get('oxidizer', 'N2O'),
            thrust_n=float(prop.get('thrust_n', 1000.0)),
            pc_bar=float(prop.get('chamber_pressure_bar', prop.get('pc_bar', 20.0))),
            mr=float(prop.get('mixture_ratio', 5.0)),
            eff_combustion=float(prop.get('combustion_efficiency', 0.95)),

            # Nozzle
            expansion_ratio=float(nozz.get('expansion_ratio', 0.0)),
            throat_diameter=float(nozz.get('throat_diameter', 0.0)),
            p_exit_bar=float(nozz.get('design_ambient_pressure_bar', 1.013)),
            L_star=float(nozz.get('L_star_mm', 1000.0)),
            contraction_ratio=float(nozz.get('contraction_ratio', 10.0)),
            theta_convergent=float(nozz.get('convergent_angle', 45.0)),
            bell_fraction=float(nozz.get('bell_fraction', 0.8)),

            # Cooling
            coolant_name=cool.get('coolant', 'REFPROP::NitrousOxide'),
            cooling_mode=cool.get('mode', 'counter-flow'),
            coolant_mass_fraction=float(cool.get('mass_fraction', 1.0)),

            coolant_p_in_bar=float(inlet.get('pressure_bar', 50.0)),
            coolant_t_in_k=float(inlet.get('temperature_k', 290.0)),

            # Channel geometry (convert mm to m)
            channel_width_throat=float(geo.get('channel_width_throat_mm', 1.0)) / 1000.0,
            channel_height=float(geo.get('channel_height_mm', 1.0)) / 1000.0,
            rib_width_throat=float(geo.get('rib_width_throat_mm', 1.0)) / 1000.0,
            wall_thickness=float(geo.get('wall_thickness_mm', 1.0)) / 1000.0,
            wall_roughness=float(geo.get('roughness_microns', 10.0)) * 1e-6,
            wall_conductivity=float(cool.get('material', {}).get('conductivity', 15.0)),
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            'engine_name': self.engine_name,
            'fuel': self.fuel,
            'oxidizer': self.oxidizer,
            'thrust_n': self.thrust_n,
            'pc_bar': self.pc_bar,
            'mr': self.mr,
            'expansion_ratio': self.expansion_ratio,
            'L_star': self.L_star,
            'coolant_name': self.coolant_name,
            'cooling_mode': self.cooling_mode,
        }
