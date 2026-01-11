"""
Configuration dataclasses for swirl injector dimensioning.

Provides structured configuration for propellant properties, 
injector geometry, and operating conditions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import json
import yaml
from pathlib import Path


@dataclass
class PropellantConfig:
    """Configuration for propellant properties."""
    fuel: str = "REFPROP::Ethanol"
    oxidizer: str = "REFPROP::NitrousOxide"
    fuel_temperature: float = 300.0  # K
    oxidizer_temperature: float = 500.0  # K

    def __post_init__(self):
        if self.fuel_temperature <= 0:
            raise ValueError("Fuel temperature must be positive")
        if self.oxidizer_temperature <= 0:
            raise ValueError("Oxidizer temperature must be positive")


@dataclass
class OperatingConditions:
    """Operating conditions for the injector."""
    inlet_pressure: float = 45e5  # Pa
    pressure_drop: float = 20e5  # Pa
    mass_flow_fuel: float = 0.20  # kg/s
    mass_flow_oxidizer: float = 0.80  # kg/s
    oxidizer_velocity: float = 100.0  # m/s (for LCSC)

    @property
    def chamber_pressure(self) -> float:
        """Calculate combustion chamber pressure."""
        return self.inlet_pressure - self.pressure_drop

    def __post_init__(self):
        if self.pressure_drop >= self.inlet_pressure:
            raise ValueError("Pressure drop must be less than inlet pressure")
        if self.mass_flow_fuel <= 0 or self.mass_flow_oxidizer <= 0:
            raise ValueError("Mass flows must be positive")


@dataclass
class GeometryConfig:
    """Geometric configuration for injector elements."""
    num_elements: int = 3
    num_fuel_ports: int = 3  # tangential fuel inlet ports per element
    num_ox_orifices: int = 1  # oxidizer inlet orifices per element
    post_thickness: float = 0.5e-3  # m
    spray_half_angle: float = 60.0  # degrees (design parameter for LCSC)
    minimum_clearance: float = 0.5e-3  # m (for manufacturing constraints)

    # Backward compatibility alias
    @property
    def num_ports(self) -> int:
        """Alias for num_fuel_ports (backward compatibility)."""
        return self.num_fuel_ports

    def __post_init__(self):
        if self.num_elements < 1 or self.num_fuel_ports < 1:
            raise ValueError("Number of elements and fuel ports must be >= 1")
        if self.num_ox_orifices < 1:
            raise ValueError("Number of oxidizer orifices must be >= 1")
        if self.post_thickness <= 0:
            raise ValueError("Post thickness must be positive")


@dataclass
class InjectorConfig:
    """Complete injector configuration."""
    propellants: PropellantConfig = field(default_factory=PropellantConfig)
    operating: OperatingConditions = field(default_factory=OperatingConditions)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "InjectorConfig":
        """Create configuration from dictionary."""
        return cls(
            propellants=PropellantConfig(**data.get("propellants", {})),
            operating=OperatingConditions(**data.get("operating", {})),
            geometry=GeometryConfig(**data.get("geometry", {}))
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InjectorConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> "InjectorConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "propellants": {
                "fuel": self.propellants.fuel,
                "oxidizer": self.propellants.oxidizer,
                "fuel_temperature": self.propellants.fuel_temperature,
                "oxidizer_temperature": self.propellants.oxidizer_temperature,
            },
            "operating": {
                "inlet_pressure": self.operating.inlet_pressure,
                "pressure_drop": self.operating.pressure_drop,
                "mass_flow_fuel": self.operating.mass_flow_fuel,
                "mass_flow_oxidizer": self.operating.mass_flow_oxidizer,
                "oxidizer_velocity": self.operating.oxidizer_velocity,
            },
            "geometry": {
                "num_elements": self.geometry.num_elements,
                "num_fuel_ports": self.geometry.num_fuel_ports,
                "num_ox_orifices": self.geometry.num_ox_orifices,
                "post_thickness": self.geometry.post_thickness,
                "spray_half_angle": self.geometry.spray_half_angle,
                "minimum_clearance": self.geometry.minimum_clearance,
            }
        }

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_json(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ColdFlowConfig:
    """Configuration for cold flow testing."""
    inlet_pressure: float = 21e5  # Pa
    pressure_drop: float = 20e5  # Pa
    ambient_temperature: float = 293.15  # K
    gas_fluid: str = "nitrogen"
    liquid_fluid: str = "water"