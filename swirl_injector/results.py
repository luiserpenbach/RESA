"""
Result dataclasses for swirl injector dimensioning.

Provides structured output for calculated properties,
geometry, and performance metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import json
from pathlib import Path


@dataclass
class FluidProperties:
    """Thermodynamic fluid properties at given conditions."""
    density: float  # kg/m³
    viscosity: float  # Pa·s
    surface_tension: Optional[float] = None  # N/m
    molar_mass: Optional[float] = None  # kg/mol
    gamma: Optional[float] = None  # isentropic expansion coefficient

    def __repr__(self) -> str:
        return (f"FluidProperties(ρ={self.density:.2f} kg/m³, "
                f"μ={self.viscosity:.2e} Pa·s)")


@dataclass
class PropellantProperties:
    """Combined fuel and oxidizer properties."""
    fuel_at_inlet: FluidProperties
    oxidizer_at_inlet: FluidProperties
    fuel_at_chamber: FluidProperties
    oxidizer_at_chamber: FluidProperties


@dataclass
class InjectorGeometry:
    """
    Calculated injector geometry dimensions.

    Fuel Side (Swirl Injector):
        - fuel_orifice_radius: Exit orifice of the swirl injector
        - fuel_port_radius: Tangential inlet port radius
        - fuel_port_length: Length of tangential inlet port (= fuel_port_radius)
        - swirl_chamber_radius: Swirl chamber radius

    Oxidizer Side:
        - ox_outlet_radius: Oxidizer outlet/injection radius (sized by target velocity)
        - ox_inlet_orifice_radius: Oxidizer inlet orifice radius (sized for choked flow)

    Mixing:
        - recess_length: Recess length for fuel/oxidizer mixing
    """
    # Fuel side (swirl injector)
    fuel_orifice_radius: float  # m - swirl injector exit orifice
    fuel_port_radius: float  # m - tangential inlet port
    swirl_chamber_radius: float  # m

    # Oxidizer side
    ox_outlet_radius: float  # m - oxidizer outlet, sized by velocity
    ox_inlet_orifice_radius: float  # m - oxidizer inlet orifice, sized for choked flow

    # Mixing geometry
    recess_length: float = 0.0  # m - recess for mixing

    # Optional dimensions
    fuel_orifice_length: Optional[float] = None  # m
    swirl_chamber_length: Optional[float] = None  # m

    @property
    def fuel_port_length(self) -> float:
        """Fuel port length (equal to port radius by convention)."""
        return self.fuel_port_radius

    # Backward compatibility aliases
    @property
    def orifice_radius(self) -> float:
        """Alias for fuel_orifice_radius (backward compatibility)."""
        return self.fuel_orifice_radius

    @property
    def port_radius(self) -> float:
        """Alias for fuel_port_radius (backward compatibility)."""
        return self.fuel_port_radius

    @property
    def oxidizer_port_radius(self) -> float:
        """Alias for ox_outlet_radius (backward compatibility)."""
        return self.ox_outlet_radius

    @property
    def oxidizer_orifice_radius(self) -> float:
        """Alias for ox_inlet_orifice_radius (backward compatibility)."""
        return self.ox_inlet_orifice_radius

    @property
    def orifice_length(self) -> Optional[float]:
        """Alias for fuel_orifice_length (backward compatibility)."""
        return self.fuel_orifice_length

    @property
    def orifice_diameter(self) -> float:
        """Fuel orifice diameter in mm."""
        return self.fuel_orifice_radius * 2000

    @property
    def port_diameter(self) -> float:
        """Fuel port diameter in mm."""
        return self.fuel_port_radius * 2000

    @property
    def swirl_chamber_diameter(self) -> float:
        """Swirl chamber diameter in mm."""
        return self.swirl_chamber_radius * 2000

    def summary(self) -> str:
        """Generate geometry summary string."""
        lines = [
            "Injector Geometry:",
            "  Fuel Side (Swirl Injector):",
            f"    Orifice radius:          {self.fuel_orifice_radius * 1000:.3f} mm",
            f"    Port radius:             {self.fuel_port_radius * 1000:.3f} mm",
            f"    Port length:             {self.fuel_port_length * 1000:.3f} mm",
            f"    Swirl chamber radius:    {self.swirl_chamber_radius * 1000:.3f} mm",
            "  Oxidizer Side:",
            f"    Outlet radius:           {self.ox_outlet_radius * 1000:.3f} mm",
            f"    Inlet orifice radius:    {self.ox_inlet_orifice_radius * 1000:.3f} mm",
            "  Mixing:",
            f"    Recess length:           {self.recess_length * 1000:.3f} mm",
        ]
        if self.fuel_orifice_length:
            lines.append(f"  Fuel orifice length:       {self.fuel_orifice_length * 1000:.3f} mm")
        if self.swirl_chamber_length:
            lines.append(f"  Swirl chamber length:      {self.swirl_chamber_length * 1000:.3f} mm")
        return "\n".join(lines)


@dataclass
class PerformanceMetrics:
    """Dimensionless performance metrics."""
    spray_half_angle: float  # degrees
    swirl_number: float
    momentum_flux_ratio: float  # J
    velocity_ratio: float
    weber_number: float
    discharge_coefficient: float
    reynolds_port: Optional[float] = None
    film_thickness: Optional[float] = None  # m
    aircore_radius: Optional[float] = None  # m

    def summary(self) -> str:
        """Generate performance summary string."""
        lines = [
            "Performance Metrics:",
            f"  Spray half angle:        {self.spray_half_angle:.2f}°",
            f"  Swirl number:            {self.swirl_number:.3f}",
            f"  Momentum flux ratio (J): {self.momentum_flux_ratio:.3f}",
            f"  Velocity ratio:          {self.velocity_ratio:.3f}",
            f"  Weber number:            {self.weber_number:.2f}",
            f"  Discharge coefficient:   {self.discharge_coefficient:.4f}",
        ]
        if self.film_thickness:
            lines.append(f"  Film thickness:          {self.film_thickness * 1000:.3f} mm")
        if self.reynolds_port:
            lines.append(f"  Reynolds (port):         {self.reynolds_port:.0f}")
        return "\n".join(lines)


@dataclass
class MassFlowResults:
    """Mass flow related results."""
    fuel_per_element: float  # kg/s
    oxidizer_per_element: float  # kg/s
    total_fuel: float  # kg/s
    total_oxidizer: float  # kg/s

    @property
    def mixture_ratio(self) -> float:
        """Oxidizer to fuel ratio."""
        return self.total_oxidizer / self.total_fuel

    def summary(self) -> str:
        """Generate mass flow summary string."""
        return (
            "Mass Flow Results:\n"
            f"  Fuel per element:     {self.fuel_per_element:.4f} kg/s\n"
            f"  Oxidizer per element: {self.oxidizer_per_element:.4f} kg/s\n"
            f"  Total fuel:           {self.total_fuel:.4f} kg/s\n"
            f"  Total oxidizer:       {self.total_oxidizer:.4f} kg/s\n"
            f"  Mixture ratio (O/F):  {self.mixture_ratio:.3f}"
        )


@dataclass
class InjectorResults:
    """Complete injector calculation results."""
    geometry: InjectorGeometry
    performance: PerformanceMetrics
    mass_flows: MassFlowResults
    propellant_properties: PropellantProperties
    injector_type: str = "LCSC"  # or "GCSC"

    def summary(self) -> str:
        """Generate complete results summary."""
        header = f"{'=' * 50}\n{self.injector_type} Injector Results\n{'=' * 50}"
        return "\n\n".join([
            header,
            self.geometry.summary(),
            self.performance.summary(),
            self.mass_flows.summary(),
        ])

    def to_dict(self) -> dict:
        """Convert results to dictionary."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def __repr__(self) -> str:
        return self.summary()


@dataclass
class ColdFlowResults:
    """Cold flow test equivalent results."""
    geometry: InjectorGeometry
    performance: PerformanceMetrics
    liquid_mass_flow: float  # kg/s
    gas_mass_flow: float  # kg/s
    gas_velocity: float  # m/s
    liquid_density: float  # kg/m³
    gas_density: float  # kg/m³

    def summary(self) -> str:
        """Generate cold flow summary string."""
        return (
            f"{'=' * 50}\n"
            f"Cold Flow Equivalent Results\n"
            f"{'=' * 50}\n\n"
            f"{self.geometry.summary()}\n\n"
            f"{self.performance.summary()}\n\n"
            f"Cold Flow Conditions:\n"
            f"  Liquid mass flow:  {self.liquid_mass_flow:.4f} kg/s\n"
            f"  Gas mass flow:     {self.gas_mass_flow:.4f} kg/s\n"
            f"  Gas velocity:      {self.gas_velocity:.2f} m/s\n"
            f"  Liquid density:    {self.liquid_density:.2f} kg/m³\n"
            f"  Gas density:       {self.gas_density:.2f} kg/m³"
        )