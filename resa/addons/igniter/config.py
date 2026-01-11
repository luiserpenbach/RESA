"""
Configuration management for torch igniter sizing tool.

Handles igniter design parameters and results with JSON serialization.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json
from pathlib import Path


@dataclass
class IgniterConfig:
    """Main configuration for igniter design.

    All pressures in Pa, temperatures in K, mass flows in kg/s.
    """

    # Operating conditions
    chamber_pressure: float        # Pa
    mixture_ratio: float           # O/F mass ratio
    total_mass_flow: float         # kg/s

    # Feed system
    ethanol_feed_pressure: float   # Pa
    n2o_feed_pressure: float       # Pa
    ethanol_feed_temperature: float  # K
    n2o_feed_temperature: float    # K

    # Design parameters
    l_star: float = 1.0           # m (characteristic length)
    expansion_ratio: float = 3.0  # Area ratio (exit/throat)
    nozzle_type: str = "conical"  # conical or bell
    conical_half_angle: float = 15.0  # degrees (if conical)

    # Injector
    n2o_orifice_count: int = 4
    ethanol_orifice_count: int = 4
    discharge_coefficient: float = 0.7

    # Environmental
    ambient_pressure: float = 101325.0  # Pa

    # Metadata
    name: str = "baseline_igniter"
    description: str = ""

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Basic validation of configuration parameters."""
        if self.chamber_pressure <= 0:
            raise ValueError("Chamber pressure must be positive")
        if self.mixture_ratio <= 0:
            raise ValueError("Mixture ratio must be positive")
        if self.total_mass_flow <= 0:
            raise ValueError("Total mass flow must be positive")
        if self.ethanol_feed_pressure <= self.chamber_pressure:
            raise ValueError("Ethanol feed pressure must exceed chamber pressure")
        if self.n2o_feed_pressure <= self.chamber_pressure:
            raise ValueError("N2O feed pressure must exceed chamber pressure")
        if self.l_star <= 0:
            raise ValueError("L* must be positive")
        if self.expansion_ratio < 1.0:
            raise ValueError("Expansion ratio must be >= 1.0")
        if self.discharge_coefficient <= 0 or self.discharge_coefficient > 1.0:
            raise ValueError("Discharge coefficient must be in (0, 1]")
        if self.n2o_orifice_count < 1:
            raise ValueError("N2O orifice count must be at least 1")
        if self.ethanol_orifice_count < 1:
            raise ValueError("Ethanol orifice count must be at least 1")
        if self.ethanol_feed_temperature <= 0:
            raise ValueError("Ethanol feed temperature must be positive")
        if self.n2o_feed_temperature <= 0:
            raise ValueError("N2O feed temperature must be positive")

    @property
    def oxidizer_mass_flow(self) -> float:
        """Mass flow of N2O oxidizer (kg/s)."""
        return self.total_mass_flow * self.mixture_ratio / (1 + self.mixture_ratio)

    @property
    def fuel_mass_flow(self) -> float:
        """Mass flow of ethanol fuel (kg/s)."""
        return self.total_mass_flow / (1 + self.mixture_ratio)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def save_json(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> 'IgniterConfig':
        """Create configuration from dictionary."""
        return cls(**data)

    @classmethod
    def load_json(cls, filepath: str) -> 'IgniterConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class IgniterResults:
    """Complete analysis results from igniter design.

    All geometric dimensions in meters, velocities in m/s,
    pressures in Pa, temperatures in K, power in W.
    """

    # Combustion properties
    flame_temperature: float       # K
    c_star: float                  # m/s
    gamma: float                   # -
    molecular_weight: float        # kg/kmol
    heat_power_output: float       # W (thermal power released)

    # Geometry
    chamber_diameter: float        # m
    chamber_length: float          # m
    chamber_volume: float          # m^3
    throat_diameter: float         # m
    throat_area: float             # m^2
    exit_diameter: float           # m
    exit_area: float               # m^2
    nozzle_length: float           # m

    # Injector
    n2o_orifice_diameter: float    # m
    ethanol_orifice_diameter: float # m
    n2o_injection_velocity: float  # m/s
    ethanol_injection_velocity: float # m/s
    n2o_pressure_drop: float       # Pa
    ethanol_pressure_drop: float   # Pa

    # Performance
    isp_theoretical: float         # s
    thrust: float                  # N

    # Mass flows (for reference)
    oxidizer_mass_flow: float      # kg/s
    fuel_mass_flow: float          # kg/s
    total_mass_flow: float         # kg/s
    mixture_ratio: float           # O/F

    # Operating conditions (for reference)
    chamber_pressure: float        # Pa

    # Optional/calculated fields
    c_star_efficiency: float = 1.0 # -

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def save_json(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> 'IgniterResults':
        """Create results from dictionary."""
        return cls(**data)

    @classmethod
    def load_json(cls, filepath: str) -> 'IgniterResults':
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def summary(self) -> str:
        """Generate text summary of key results."""
        lines = [
            "=" * 60,
            "TORCH IGNITER DESIGN SUMMARY",
            "=" * 60,
            "",
            "OPERATING CONDITIONS:",
            f"  Chamber Pressure:     {self.chamber_pressure/1e5:.2f} bar",
            f"  Mixture Ratio (O/F):  {self.mixture_ratio:.2f}",
            f"  Total Mass Flow:      {self.total_mass_flow*1000:.2f} g/s",
            f"  N2O Mass Flow:        {self.oxidizer_mass_flow*1000:.2f} g/s",
            f"  Ethanol Mass Flow:    {self.fuel_mass_flow*1000:.2f} g/s",
            "",
            "COMBUSTION:",
            f"  Flame Temperature:    {self.flame_temperature:.0f} K",
            f"  C* (theoretical):     {self.c_star:.1f} m/s",
            f"  Gamma:                {self.gamma:.3f}",
            f"  Molecular Weight:     {self.molecular_weight:.2f} kg/kmol",
            f"  Heat Power Output:    {self.heat_power_output/1000:.2f} kW",
            "",
            "GEOMETRY:",
            f"  Chamber Diameter:     {self.chamber_diameter*1000:.2f} mm",
            f"  Chamber Length:       {self.chamber_length*1000:.2f} mm",
            f"  Chamber Volume:       {self.chamber_volume*1e6:.2f} cm3",
            f"  Throat Diameter:      {self.throat_diameter*1000:.2f} mm",
            f"  Exit Diameter:        {self.exit_diameter*1000:.2f} mm",
            f"  Nozzle Length:        {self.nozzle_length*1000:.2f} mm",
            "",
            "INJECTOR:",
            f"  N2O Orifice Diameter: {self.n2o_orifice_diameter*1000:.3f} mm",
            f"  EtOH Orifice Diameter:{self.ethanol_orifice_diameter*1000:.3f} mm",
            f"  N2O Injection Vel:    {self.n2o_injection_velocity:.1f} m/s",
            f"  EtOH Injection Vel:   {self.ethanol_injection_velocity:.1f} m/s",
            f"  N2O Pressure Drop:    {self.n2o_pressure_drop/1e5:.2f} bar",
            f"  EtOH Pressure Drop:   {self.ethanol_pressure_drop/1e5:.2f} bar",
            "",
            "PERFORMANCE:",
            f"  Theoretical Isp:      {self.isp_theoretical:.1f} s",
            f"  Thrust:               {self.thrust:.2f} N",
            "=" * 60,
        ]
        return "\n".join(lines)
