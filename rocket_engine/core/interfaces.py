"""
Core interfaces and protocols for RESA.
These define contracts that implementations must follow.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Dict, Any, Optional
import numpy as np


# =============================================================================
# RESULT PROTOCOLS
# =============================================================================

class HasPerformance(Protocol):
    """Protocol for objects that have performance metrics."""
    isp_vac: float
    isp_sea: float
    thrust_vac: float
    thrust_sea: float


class HasGeometry(Protocol):
    """Protocol for objects that have geometric data."""
    dt_mm: float  # Throat diameter
    de_mm: float  # Exit diameter
    length_mm: float


# =============================================================================
# SOLVER INTERFACES
# =============================================================================

class BaseSolver(ABC):
    """Abstract base class for all solvers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable solver name."""
        pass
    
    @abstractmethod
    def validate_inputs(self) -> bool:
        """Validate that inputs are physically reasonable."""
        pass


class CombustionSolverInterface(ABC):
    """Interface for combustion analysis solvers (CEA, Cantera, etc.)."""
    
    @abstractmethod
    def run(self, pc_bar: float, mr: float, eps: float) -> 'CombustionResult':
        """Run combustion analysis at given conditions."""
        pass
    
    @abstractmethod
    def get_cstar(self, pc_bar: float, mr: float) -> float:
        """Get characteristic velocity."""
        pass


class CoolingSolverInterface(ABC):
    """Interface for cooling analysis solvers."""
    
    @abstractmethod
    def solve(self, 
              mdot_coolant: float,
              p_in: float,
              t_in: float,
              T_gas: np.ndarray,
              h_gas: np.ndarray) -> Dict[str, np.ndarray]:
        """Solve cooling problem."""
        pass


class GeometryGeneratorInterface(ABC):
    """Interface for geometry generators."""
    
    @abstractmethod
    def generate(self, **kwargs) -> 'GeometryData':
        """Generate geometry."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Check if geometry is manufacturable."""
        pass


# =============================================================================
# COMPONENT INTERFACES
# =============================================================================

class FlowComponent(ABC):
    """Base class for flow components (injectors, valves, orifices)."""
    
    @abstractmethod
    def get_mass_flow(self, p_upstream: float, p_downstream: float, 
                      T: float, rho: float) -> float:
        """Calculate mass flow through component."""
        pass
    
    @abstractmethod
    def get_pressure_drop(self, mdot: float, rho: float) -> float:
        """Calculate pressure drop for given mass flow."""
        pass


class InjectorInterface(FlowComponent):
    """Interface for injector elements."""
    
    @property
    @abstractmethod
    def cd(self) -> float:
        """Discharge coefficient."""
        pass
    
    @property
    @abstractmethod
    def area(self) -> float:
        """Total flow area [m²]."""
        pass


# =============================================================================
# ANALYSIS INTERFACES  
# =============================================================================

class AnalysisModule(ABC):
    """Base class for analysis modules that can be plugged into the UI."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Module name for UI display."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of what this module does."""
        pass
    
    @property
    @abstractmethod
    def icon(self) -> str:
        """Emoji or icon for UI."""
        pass
    
    @abstractmethod
    def render_inputs(self, st) -> Dict[str, Any]:
        """Render Streamlit input widgets, return collected inputs."""
        pass
    
    @abstractmethod
    def run_analysis(self, inputs: Dict[str, Any]) -> Any:
        """Run the analysis with given inputs."""
        pass
    
    @abstractmethod
    def render_results(self, st, results: Any) -> None:
        """Render results in Streamlit."""
        pass


# =============================================================================
# FLUID INTERFACE
# =============================================================================

@dataclass
class FluidState:
    """Immutable snapshot of fluid thermodynamic state."""
    pressure: float      # Pa
    temperature: float   # K
    enthalpy: float      # J/kg
    density: float       # kg/m³
    specific_heat: float # J/(kg·K)
    viscosity: float     # Pa·s
    conductivity: float  # W/(m·K)
    quality: float       # -1 if single phase, 0-1 if two-phase
    
    @property
    def prandtl(self) -> float:
        if self.conductivity > 0:
            return (self.specific_heat * self.viscosity) / self.conductivity
        return 0.7  # Default
    
    @property
    def kinematic_viscosity(self) -> float:
        return self.viscosity / self.density
    
    def __post_init__(self):
        # Validate physical bounds
        if self.pressure <= 0:
            raise ValueError(f"Pressure must be positive, got {self.pressure}")
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {self.temperature}")
        if self.density <= 0:
            raise ValueError(f"Density must be positive, got {self.density}")


class FluidInterface(ABC):
    """Interface for fluid property providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Fluid name."""
        pass
    
    @abstractmethod
    def get_state(self, pressure: float, 
                  temperature: Optional[float] = None,
                  enthalpy: Optional[float] = None) -> FluidState:
        """Get complete fluid state at given conditions."""
        pass
    
    @abstractmethod
    def get_saturation_pressure(self, temperature: float) -> float:
        """Get vapor pressure at given temperature."""
        pass
    
    @abstractmethod
    def get_saturation_temperature(self, pressure: float) -> float:
        """Get saturation temperature at given pressure."""
        pass
