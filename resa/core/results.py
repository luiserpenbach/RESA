"""
Result dataclasses for RESA.
All analysis results are immutable dataclasses with clear structure.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
import numpy as np
from datetime import datetime


# =============================================================================
# COMBUSTION RESULTS
# =============================================================================

@dataclass(frozen=True)
class CombustionResult:
    """Result from combustion analysis (CEA)."""
    pc_bar: float           # Chamber pressure
    mr: float               # Mixture ratio (O/F)
    cstar: float            # Characteristic velocity [m/s]
    isp_vac: float          # Vacuum specific impulse [s]
    isp_opt: float          # Optimized/ambient Isp [s]
    T_combustion: float     # Combustion temperature [K]
    gamma: float            # Specific heat ratio
    mw: float               # Molecular weight [kg/kmol]
    mach_exit: float        # Exit Mach number
    
    @property
    def R_specific(self) -> float:
        """Specific gas constant [J/(kg·K)]."""
        return 8314.46 / self.mw


@dataclass(frozen=True)
class NozzleGeometry:
    """Nozzle contour geometry."""
    x_full: np.ndarray      # Axial coordinates [mm]
    y_full: np.ndarray      # Radial coordinates [mm]
    
    # Section breakdown (for detailed analysis)
    x_chamber: np.ndarray
    y_chamber: np.ndarray
    x_convergent: np.ndarray
    y_convergent: np.ndarray
    x_throat: np.ndarray
    y_throat: np.ndarray
    x_divergent: np.ndarray
    y_divergent: np.ndarray
    
    @property
    def throat_radius(self) -> float:
        """Throat radius [mm]."""
        return float(np.min(self.y_full))
    
    @property
    def exit_radius(self) -> float:
        """Exit radius [mm]."""
        return float(self.y_full[-1])
    
    @property
    def expansion_ratio(self) -> float:
        """Area expansion ratio."""
        return (self.exit_radius / self.throat_radius) ** 2
    
    @property
    def total_length(self) -> float:
        """Total length [mm]."""
        return float(self.x_full[-1] - self.x_full[0])
    
    def __hash__(self):
        return hash((self.throat_radius, self.exit_radius, self.total_length))


@dataclass(frozen=True)
class CoolingChannelGeometry:
    """Cooling channel geometry at all stations."""
    x_contour: np.ndarray           # Axial position [m]
    radius_contour: np.ndarray      # Inner chamber radius [m]
    channel_width: np.ndarray       # [m]
    channel_height: np.ndarray      # [m]
    rib_width: np.ndarray           # [m]
    wall_thickness: np.ndarray      # [m]
    roughness: float                # Surface roughness [m]
    number_of_channels: int
    helix_angle: np.ndarray = None  # [rad]
    
    @property
    def hydraulic_diameter(self) -> np.ndarray:
        """Hydraulic diameter [m]."""
        return 2.0 * self.channel_width * self.channel_height / (
            self.channel_width + self.channel_height
        )
    
    @property
    def flow_area(self) -> np.ndarray:
        """Flow area per channel [m²]."""
        return self.channel_width * self.channel_height
    
    def __hash__(self):
        return hash((self.number_of_channels, self.roughness, len(self.x_contour)))


@dataclass
class CoolingResult:
    """Result from regenerative cooling analysis."""
    T_coolant: np.ndarray       # Bulk coolant temperature [K]
    P_coolant: np.ndarray       # Coolant pressure [Pa]
    T_wall_hot: np.ndarray      # Hot gas side wall temperature [K]
    T_wall_cold: np.ndarray     # Coolant side wall temperature [K]
    q_flux: np.ndarray          # Heat flux [W/m²]
    velocity: np.ndarray        # Coolant velocity [m/s]
    density: np.ndarray         # Coolant density [kg/m³]
    quality: np.ndarray         # Vapor quality (-1 if single phase)
    
    @property
    def max_wall_temp(self) -> float:
        return float(np.max(self.T_wall_hot))
    
    @property
    def max_heat_flux(self) -> float:
        return float(np.max(self.q_flux))
    
    @property
    def pressure_drop(self) -> float:
        """Total pressure drop [Pa]."""
        return float(np.max(self.P_coolant) - np.min(self.P_coolant))
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for legacy compatibility."""
        return {
            'T_coolant': self.T_coolant,
            'P_coolant': self.P_coolant,
            'T_wall_hot': self.T_wall_hot,
            'T_wall_cold': self.T_wall_cold,
            'q_flux': self.q_flux,
            'velocity': self.velocity,
            'density': self.density,
            'quality': self.quality
        }


@dataclass
class EngineDesignResult:
    """Complete result from engine design/analysis."""
    # Identification
    timestamp: datetime = field(default_factory=datetime.now)
    run_type: str = "design"  # "design", "off_design", "throttle"
    
    # Operating Point
    pc_bar: float = 0.0
    mr: float = 0.0
    
    # Performance
    isp_vac: float = 0.0
    isp_sea: float = 0.0
    thrust_vac: float = 0.0
    thrust_sea: float = 0.0
    massflow_total: float = 0.0
    massflow_ox: float = 0.0
    massflow_fuel: float = 0.0
    
    # Key Dimensions
    dt_mm: float = 0.0
    de_mm: float = 0.0
    length_mm: float = 0.0
    expansion_ratio: float = 0.0
    
    # Combustion Data
    combustion: Optional[CombustionResult] = None
    
    # Geometry
    nozzle_geometry: Optional[NozzleGeometry] = None
    channel_geometry: Optional[CoolingChannelGeometry] = None
    
    # Thermal Analysis
    cooling: Optional[CoolingResult] = None
    
    # Gas Dynamics (1D arrays along nozzle)
    mach_numbers: Optional[np.ndarray] = None
    T_gas_recovery: Optional[np.ndarray] = None
    h_gas: Optional[np.ndarray] = None
    
    # Warnings/Notes
    warnings: List[str] = field(default_factory=list)
    
    @property
    def cooling_data(self) -> Dict[str, np.ndarray]:
        """Legacy compatibility for cooling_data dict."""
        if self.cooling:
            return self.cooling.to_dict()
        return {}
    
    @property
    def geometry(self):
        """Legacy compatibility."""
        return self.nozzle_geometry
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
    
    def summary_dict(self) -> Dict[str, Any]:
        """Return key metrics as dictionary for display."""
        return {
            "Thrust (Vacuum)": f"{self.thrust_vac:.1f} N",
            "Thrust (Sea Level)": f"{self.thrust_sea:.1f} N",
            "Isp (Vacuum)": f"{self.isp_vac:.1f} s",
            "Isp (Sea Level)": f"{self.isp_sea:.1f} s",
            "Chamber Pressure": f"{self.pc_bar:.1f} bar",
            "Mixture Ratio": f"{self.mr:.2f}",
            "Throat Diameter": f"{self.dt_mm:.2f} mm",
            "Exit Diameter": f"{self.de_mm:.2f} mm",
            "Expansion Ratio": f"{self.expansion_ratio:.2f}",
            "Max Wall Temp": f"{self.cooling.max_wall_temp:.0f} K" if self.cooling else "N/A"
        }


# =============================================================================
# INJECTOR RESULTS
# =============================================================================

@dataclass
class InjectorGeometryResult:
    """Result from injector sizing."""
    # Swirl Chamber
    orifice_diameter_mm: float
    chamber_diameter_mm: float
    inlet_port_diameter_mm: float
    num_inlet_ports: int
    chamber_length_mm: float
    orifice_length_mm: float
    
    # Performance Estimates
    discharge_coefficient: float
    spray_half_angle_deg: float
    film_thickness_mm: float
    swirl_number: float
    
    # For Coaxial Injectors
    oxidizer_annulus_od_mm: Optional[float] = None
    oxidizer_annulus_id_mm: Optional[float] = None
    
    @property
    def orifice_area_mm2(self) -> float:
        return np.pi * (self.orifice_diameter_mm / 2) ** 2


@dataclass
class InjectorPerformanceResult:
    """Injector operating performance."""
    mass_flow_kg_s: float
    pressure_drop_bar: float
    velocity_m_s: float
    momentum_flux_ratio: float  # J
    weber_number: float
    reynolds_number: float


# =============================================================================
# THROTTLE ANALYSIS RESULTS
# =============================================================================

@dataclass
class ThrottlePoint:
    """Single point on a throttle curve."""
    throttle_pct: float
    pc_bar: float
    mr: float
    thrust_n: float
    isp_s: float
    max_wall_temp_k: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'throttle_pct': self.throttle_pct,
            'pc_bar': self.pc_bar,
            'mr': self.mr,
            'thrust_n': self.thrust_n,
            'isp_s': self.isp_s,
            'max_wall_temp_k': self.max_wall_temp_k
        }


@dataclass
class ThrottleCurve:
    """Complete throttle analysis result."""
    points: List[ThrottlePoint]
    throttle_mode: str  # "ox_only", "both", "fuel_only"
    fuel_control: str   # "venturi", "orifice"
    
    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([p.to_dict() for p in self.points])
    
    @property
    def min_thrust(self) -> float:
        return min(p.thrust_n for p in self.points)
    
    @property
    def max_thrust(self) -> float:
        return max(p.thrust_n for p in self.points)
    
    @property
    def throttle_ratio(self) -> float:
        return self.max_thrust / self.min_thrust if self.min_thrust > 0 else float('inf')
