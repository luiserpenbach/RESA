"""
Configuration classes for RESA with validation and YAML support.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Literal
from pathlib import Path
import yaml
import json


# =============================================================================
# PROPELLANT DEFINITIONS
# =============================================================================

PROPELLANT_ALIASES = {
    # Oxidizers
    "N2O": "NitrousOxide",
    "Nitrous": "NitrousOxide",
    "LOX": "Oxygen",
    "O2": "Oxygen",
    "H2O2": "HydrogenPeroxide",
    
    # Fuels
    "Ethanol": "Ethanol",
    "Ethanol90": "Ethanol[0.866]&Water[0.134]",
    "Ethanol80": "Ethanol[0.735]&Water[0.265]",
    "RP1": "n-Dodecane",
    "Kerosene": "n-Dodecane",
    "Methane": "Methane",
    "LCH4": "Methane",
    "IPA": "Isopropanol",
    "Propane": "Propane",
}

# Default material properties (W/m·K)
MATERIAL_CONDUCTIVITY = {
    "copper": 385.0,
    "inconel718": 11.4,
    "inconel625": 9.8,
    "stainless316": 16.3,
    "aluminum6061": 167.0,
    "haynes230": 8.9,
}


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    
    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)


# =============================================================================
# ENGINE CONFIGURATION
# =============================================================================

@dataclass
class EngineConfig:
    """
    Complete engine configuration with validation.
    
    All units are SI unless otherwise noted in the field name:
    - Pressures: bar (for user convenience)
    - Temperatures: K
    - Lengths: m (except explicit _mm fields)
    - Mass flow: kg/s
    """
    
    # === IDENTIFICATION ===
    engine_name: str = "Unnamed Engine"
    version: str = "1.0"
    designer: str = ""
    
    # === PROPELLANTS ===
    fuel: str = "Ethanol90"
    oxidizer: str = "N2O"
    
    # === PERFORMANCE TARGETS ===
    thrust_n: float = 1000.0
    pc_bar: float = 20.0
    mr: float = 4.0
    eff_combustion: float = 0.95
    
    # === NOZZLE DESIGN ===
    throat_diameter: float = 0.0    # [m] If 0, calculated from thrust
    expansion_ratio: float = 0.0    # If 0, calculated for optimal at p_exit
    p_exit_bar: float = 1.013       # Design exit pressure
    L_star: float = 1100.0          # [mm]
    contraction_ratio: float = 10.0
    theta_convergent: float = 30.0  # [deg]
    bell_fraction: float = 0.8
    
    # === COOLING SYSTEM ===
    coolant_name: str = "REFPROP::NitrousOxide"
    cooling_mode: Literal['counter-flow', 'co-flow'] = 'counter-flow'
    coolant_mass_fraction: float = 1.0
    
    # Coolant inlet conditions
    coolant_p_in_bar: float = 50.0
    coolant_t_in_k: float = 290.0
    
    # Channel geometry at throat
    channel_width_throat: float = 1.0e-3   # [m]
    channel_height: float = 0.75e-3        # [m]
    rib_width_throat: float = 0.6e-3       # [m]
    
    # Wall properties
    wall_thickness: float = 0.5e-3         # [m]
    wall_roughness: float = 20e-6          # [m]
    wall_conductivity: float = 15.0        # [W/(m·K)]
    wall_material: str = "inconel718"
    
    # === INJECTOR (Optional) ===
    injector_dp_bar: float = 0.0           # If 0, use 15% of Pc
    
    def __post_init__(self):
        """Resolve aliases and set defaults."""
        # Set default injector pressure drop
        if self.injector_dp_bar <= 0:
            self.injector_dp_bar = 0.15 * self.pc_bar
        
        # Set wall conductivity from material if specified
        if self.wall_material.lower() in MATERIAL_CONDUCTIVITY:
            self.wall_conductivity = MATERIAL_CONDUCTIVITY[self.wall_material.lower()]
    
    def validate(self) -> ValidationResult:
        """Validate configuration for physical reasonableness."""
        result = ValidationResult(is_valid=True)
        
        # === PRESSURE CHECKS ===
        if self.pc_bar <= 0:
            result.add_error(f"Chamber pressure must be positive (got {self.pc_bar})")
        if self.pc_bar > 300:
            result.add_warning(f"Chamber pressure {self.pc_bar} bar is very high")
        
        if self.coolant_p_in_bar <= self.pc_bar:
            result.add_error(
                f"Coolant inlet pressure ({self.coolant_p_in_bar} bar) must exceed "
                f"chamber pressure ({self.pc_bar} bar)"
            )
        
        # === GEOMETRY CHECKS ===
        if self.contraction_ratio < 2:
            result.add_error(f"Contraction ratio {self.contraction_ratio} too low (<2)")
        if self.contraction_ratio > 20:
            result.add_warning(f"Contraction ratio {self.contraction_ratio} is high (>20)")
        
        if self.L_star < 500:
            result.add_warning(f"L* of {self.L_star} mm is low for stable combustion")
        if self.L_star > 2500:
            result.add_warning(f"L* of {self.L_star} mm is high, consider reducing")
        
        if self.expansion_ratio > 0 and self.expansion_ratio < 1.5:
            result.add_error(f"Expansion ratio {self.expansion_ratio} too low")
        
        # === COOLING CHECKS ===
        if self.channel_height < 0.3e-3:
            result.add_warning(f"Channel height {self.channel_height*1000:.2f} mm very small")
        if self.wall_thickness < 0.3e-3:
            result.add_warning(f"Wall thickness {self.wall_thickness*1000:.2f} mm very thin")
        
        if self.coolant_t_in_k < 200:
            result.add_warning(f"Coolant inlet temp {self.coolant_t_in_k} K is cryogenic")
        
        # === PERFORMANCE CHECKS ===
        if self.mr < 1:
            result.add_warning(f"Mixture ratio {self.mr} is fuel-rich")
        if self.mr > 10:
            result.add_warning(f"Mixture ratio {self.mr} is very oxidizer-rich")
        
        if self.eff_combustion < 0.8:
            result.add_warning(f"Combustion efficiency {self.eff_combustion} is low")
        if self.eff_combustion > 1.0:
            result.add_error("Combustion efficiency cannot exceed 1.0")
        
        return result
    
    @classmethod
    def from_yaml(cls, path: str) -> 'EngineConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls._from_nested_dict(data)
    
    @classmethod
    def _from_nested_dict(cls, data: Dict[str, Any]) -> 'EngineConfig':
        """Parse potentially nested YAML structure."""
        
        def get(section: str, key: str, default=None):
            if section in data:
                return data[section].get(key, default)
            return data.get(key, default)
        
        # Handle both flat and nested formats
        meta = data.get('meta', {})
        prop = data.get('propulsion', data)
        nozz = data.get('nozzle', data)
        cool = data.get('cooling', {})
        geo = cool.get('geometry', data)
        inlet = cool.get('inlet', data)
        
        return cls(
            engine_name=meta.get('engine_name', data.get('engine_name', 'Unnamed')),
            version=meta.get('version', '1.0'),
            designer=meta.get('designer', ''),
            
            fuel=prop.get('fuel', 'Ethanol90'),
            oxidizer=prop.get('oxidizer', 'N2O'),
            thrust_n=float(prop.get('thrust_n', 1000)),
            pc_bar=float(prop.get('pc_bar', prop.get('chamber_pressure_bar', 20))),
            mr=float(prop.get('mr', prop.get('mixture_ratio', 4))),
            eff_combustion=float(prop.get('eff_combustion', 
                                         prop.get('combustion_efficiency', 0.95))),
            
            throat_diameter=float(nozz.get('throat_diameter', 0)),
            expansion_ratio=float(nozz.get('expansion_ratio', 0)),
            p_exit_bar=float(nozz.get('design_ambient_pressure_bar', 
                                     nozz.get('p_exit_bar', 1.013))),
            L_star=float(nozz.get('L_star_mm', nozz.get('L_star', 1100))),
            contraction_ratio=float(nozz.get('contraction_ratio', 10)),
            theta_convergent=float(nozz.get('convergent_angle', 
                                           nozz.get('theta_convergent', 30))),
            bell_fraction=float(nozz.get('bell_fraction', 0.8)),
            
            coolant_name=cool.get('coolant', cool.get('coolant_name', 'REFPROP::NitrousOxide')),
            cooling_mode=cool.get('mode', 'counter-flow'),
            coolant_mass_fraction=float(cool.get('mass_fraction', 1.0)),
            coolant_p_in_bar=float(inlet.get('pressure_bar', 
                                            inlet.get('coolant_p_in_bar', 50))),
            coolant_t_in_k=float(inlet.get('temperature_k', 
                                          inlet.get('coolant_t_in_k', 290))),
            
            channel_width_throat=float(geo.get('channel_width_throat_mm', 1.0)) / 1000
                if 'channel_width_throat_mm' in geo 
                else float(geo.get('channel_width_throat', 1e-3)),
            channel_height=float(geo.get('channel_height_mm', 0.75)) / 1000
                if 'channel_height_mm' in geo
                else float(geo.get('channel_height', 0.75e-3)),
            rib_width_throat=float(geo.get('rib_width_throat_mm', 0.6)) / 1000
                if 'rib_width_throat_mm' in geo
                else float(geo.get('rib_width_throat', 0.6e-3)),
            wall_thickness=float(geo.get('wall_thickness_mm', 0.5)) / 1000
                if 'wall_thickness_mm' in geo
                else float(geo.get('wall_thickness', 0.5e-3)),
            wall_roughness=float(geo.get('roughness_microns', 20)) * 1e-6
                if 'roughness_microns' in geo
                else float(geo.get('wall_roughness', 20e-6)),
            wall_conductivity=float(cool.get('material', {}).get('conductivity', 15)),
        )
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        data = {
            'meta': {
                'engine_name': self.engine_name,
                'version': self.version,
                'designer': self.designer,
            },
            'propulsion': {
                'fuel': self.fuel,
                'oxidizer': self.oxidizer,
                'thrust_n': self.thrust_n,
                'pc_bar': self.pc_bar,
                'mr': self.mr,
                'eff_combustion': self.eff_combustion,
            },
            'nozzle': {
                'throat_diameter': self.throat_diameter,
                'expansion_ratio': self.expansion_ratio,
                'p_exit_bar': self.p_exit_bar,
                'L_star_mm': self.L_star,
                'contraction_ratio': self.contraction_ratio,
                'theta_convergent': self.theta_convergent,
                'bell_fraction': self.bell_fraction,
            },
            'cooling': {
                'coolant': self.coolant_name,
                'mode': self.cooling_mode,
                'mass_fraction': self.coolant_mass_fraction,
                'inlet': {
                    'pressure_bar': self.coolant_p_in_bar,
                    'temperature_k': self.coolant_t_in_k,
                },
                'geometry': {
                    'channel_width_throat_mm': self.channel_width_throat * 1000,
                    'channel_height_mm': self.channel_height * 1000,
                    'rib_width_throat_mm': self.rib_width_throat * 1000,
                    'wall_thickness_mm': self.wall_thickness * 1000,
                    'roughness_microns': self.wall_roughness * 1e6,
                },
                'material': {
                    'name': self.wall_material,
                    'conductivity': self.wall_conductivity,
                }
            }
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary."""
        return asdict(self)
    
    def copy(self, **updates) -> 'EngineConfig':
        """Create a copy with optional field updates."""
        data = asdict(self)
        data.update(updates)
        return EngineConfig(**data)


# =============================================================================
# ANALYSIS PRESETS
# =============================================================================

@dataclass
class AnalysisPreset:
    """Predefined analysis configurations."""
    name: str
    description: str
    config: EngineConfig
    
    @classmethod
    def demo_50n(cls) -> 'AnalysisPreset':
        """Small demonstrator engine."""
        return cls(
            name="Demo 50N",
            description="Small test engine for concept validation",
            config=EngineConfig(
                engine_name="Demo-50N",
                fuel="C2H6",
                oxidizer="N2O",
                thrust_n=50,
                pc_bar=7,
                mr=6.0,
                L_star=1200,
                contraction_ratio=15,
            )
        )
    
    @classmethod
    def hopper_2kn(cls) -> 'AnalysisPreset':
        """2kN Ethanol/N2O hopper engine."""
        return cls(
            name="Hopper 2kN",
            description="2.2 kN regeneratively cooled landing engine",
            config=EngineConfig(
                engine_name="Hopper E2-1A",
                fuel="Ethanol90",
                oxidizer="N2O",
                thrust_n=2200,
                pc_bar=25,
                mr=4.0,
                L_star=1200,
                contraction_ratio=12,
                expansion_ratio=4.1,
                coolant_p_in_bar=97,
                coolant_t_in_k=298,
            )
        )
