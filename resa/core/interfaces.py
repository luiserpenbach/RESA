"""
Core interfaces and protocols for RESA.

These define contracts that all implementations must follow, enabling:
- Dependency injection for testing
- Swappable solver implementations
- Clear API contracts
- Plugin architecture for add-ons
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Dict, Any, Optional, List, TypeVar, Generic
import numpy as np
from datetime import datetime


# Type variable for generic solver outputs
T = TypeVar('T')


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
    dt_mm: float
    de_mm: float
    length_mm: float


# =============================================================================
# SOLVER INTERFACES
# =============================================================================

class Solver(ABC, Generic[T]):
    """
    Base interface for all physics solvers.

    Type parameter T indicates the result type.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable solver name."""
        pass

    @abstractmethod
    def solve(self, *args, **kwargs) -> T:
        """Execute the solver and return typed results."""
        pass

    @abstractmethod
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate inputs before solving."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Return solver metadata for logging/debugging."""
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
        }


class CombustionSolver(Solver):
    """Interface for combustion analysis solvers (CEA, Cantera, etc.)."""

    @abstractmethod
    def run(self, pc_bar: float, mr: float, eps: float) -> 'CombustionResult':
        """Run combustion analysis at given conditions."""
        pass

    @abstractmethod
    def get_cstar(self, pc_bar: float, mr: float) -> float:
        """Get characteristic velocity at conditions."""
        pass

    @abstractmethod
    def get_gamma(self, pc_bar: float, mr: float) -> float:
        """Get specific heat ratio at conditions."""
        pass


class CoolingSolver(Solver):
    """Interface for regenerative cooling analysis solvers."""

    @abstractmethod
    def solve(self,
              mdot_coolant: float,
              p_in: float,
              t_in: float,
              geometry: Any,
              T_gas: np.ndarray,
              h_gas: np.ndarray,
              mode: str = 'counter-flow') -> Any:
        """Solve the cooling problem along the channel."""
        pass


# =============================================================================
# GEOMETRY INTERFACES
# =============================================================================

class GeometryGenerator(ABC):
    """Base interface for geometry generators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Generator name."""
        pass

    @abstractmethod
    def generate(self, **params) -> Any:
        """Generate geometry from parameters."""
        pass

    @abstractmethod
    def validate_parameters(self, **params) -> bool:
        """Validate parameters before generation."""
        pass

    def export_dxf(self, geometry: Any, filepath: str) -> None:
        """Optional: Export to DXF format."""
        raise NotImplementedError("DXF export not implemented")

    def export_stl(self, geometry: Any, filepath: str) -> None:
        """Optional: Export to STL format."""
        raise NotImplementedError("STL export not implemented")


# =============================================================================
# FLUID INTERFACES
# =============================================================================

@dataclass
class FluidState:
    """Represents thermodynamic state of a fluid."""
    pressure: float      # Pa
    temperature: float   # K
    density: float       # kg/m³
    enthalpy: float      # J/kg
    entropy: float       # J/(kg·K)
    quality: float       # Vapor quality (0-1), -1 for supercritical
    phase: str           # 'liquid', 'vapor', 'two-phase', 'supercritical'
    viscosity: Optional[float] = None     # Pa·s
    conductivity: Optional[float] = None  # W/(m·K)
    cp: Optional[float] = None            # J/(kg·K)

    @property
    def prandtl(self) -> float:
        """Prandtl number."""
        if self.conductivity and self.cp and self.conductivity > 0:
            return (self.cp * self.viscosity) / self.conductivity
        return 0.7

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity [m²/s]."""
        if self.viscosity and self.density > 0:
            return self.viscosity / self.density
        return 0.0


class FluidProvider(ABC):
    """Base interface for fluid property providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Fluid name."""
        pass

    @property
    @abstractmethod
    def critical_point(self) -> Dict[str, float]:
        """Return {'T_crit': K, 'P_crit': Pa, 'rho_crit': kg/m³}."""
        pass

    @abstractmethod
    def get_state(self, pressure: float, temperature: float) -> FluidState:
        """Get fluid state at given P, T."""
        pass

    @abstractmethod
    def get_state_ph(self, pressure: float, enthalpy: float) -> FluidState:
        """Get fluid state at given P, H."""
        pass

    @abstractmethod
    def get_saturation_pressure(self, temperature: float) -> float:
        """Get vapor pressure at given temperature [Pa]."""
        pass

    @abstractmethod
    def get_saturation_temperature(self, pressure: float) -> float:
        """Get saturation temperature at given pressure [K]."""
        pass

    @abstractmethod
    def get_transport_properties(self, state: FluidState) -> Dict[str, float]:
        """Get transport properties at state."""
        pass


# =============================================================================
# VISUALIZATION INTERFACES
# =============================================================================

class Plotter(ABC):
    """Base interface for all Plotly-based plotters."""

    @abstractmethod
    def create_figure(self, data: Any, **options) -> Any:
        """Create a Plotly figure from input data."""
        pass

    @abstractmethod
    def to_html(self, figure: Any, **options) -> str:
        """Export figure to embeddable HTML string."""
        pass

    def to_json(self, figure: Any) -> str:
        """Export figure to JSON for web embedding."""
        return figure.to_json()

    def show(self, figure: Any) -> None:
        """Display figure interactively."""
        figure.show()


class Viewer3D(ABC):
    """Interface for 3D WebGL visualization components."""

    @abstractmethod
    def render(self, geometry: Any, **options) -> Any:
        """Render 3D geometry."""
        pass

    @abstractmethod
    def to_html(self, **options) -> str:
        """Export to standalone HTML with WebGL."""
        pass

    @abstractmethod
    def export_gltf(self, filepath: str) -> None:
        """Export to GLTF format for external viewers."""
        pass


# =============================================================================
# REPORT INTERFACES
# =============================================================================

class ReportGenerator(ABC):
    """Base interface for report generation."""

    @abstractmethod
    def generate(self, result: Any, output_path: Optional[str] = None,
                 **options) -> str:
        """Generate report and return HTML/path."""
        pass

    @abstractmethod
    def add_section(self, title: str, content: str, order: int = 0) -> None:
        """Add a custom section to the report."""
        pass

    def get_supported_formats(self) -> List[str]:
        """Return list of supported output formats."""
        return ['html']


# =============================================================================
# ANALYSIS MODULE INTERFACE (FOR ADD-ONS)
# =============================================================================

class AnalysisModule(ABC):
    """
    Base class for analysis modules that can be plugged into the UI.

    This enables add-on modules like igniter sizing, injector design, etc.
    to integrate seamlessly with the main application.
    """

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
        """Emoji or icon string for UI."""
        pass

    @property
    def category(self) -> str:
        """Category for grouping in UI ('design', 'analysis', 'addon')."""
        return "addon"

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
# VERSION CONTROL INTERFACES
# =============================================================================

@dataclass
class DesignVersion:
    """Represents a version of a design in version control."""
    version_id: str
    timestamp: datetime
    config_hash: str
    description: str
    author: str
    parent_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class VersionControl(ABC):
    """Interface for design version control."""

    @abstractmethod
    def save_version(self, config: Any, result: Any,
                     description: str, author: str) -> DesignVersion:
        """Save a new version of the design."""
        pass

    @abstractmethod
    def load_version(self, version_id: str) -> tuple:
        """Load config and result from a version."""
        pass

    @abstractmethod
    def list_versions(self, project_name: str) -> List[DesignVersion]:
        """List all versions of a project."""
        pass

    @abstractmethod
    def diff_versions(self, v1_id: str, v2_id: str) -> Dict[str, Any]:
        """Compare two versions."""
        pass

    @abstractmethod
    def tag_version(self, version_id: str, tag: str) -> None:
        """Add a tag to a version (e.g., 'release', 'baseline')."""
        pass


# =============================================================================
# OUTPUT MANAGEMENT
# =============================================================================

@dataclass
class OutputConfig:
    """Configuration for output management."""
    base_dir: str = "./output"
    create_project_dirs: bool = True
    timestamp_files: bool = True
    default_formats: List[str] = field(default_factory=lambda: ['html', 'csv'])


class OutputManager(ABC):
    """Interface for managing output files and directories."""

    @abstractmethod
    def get_output_dir(self, project_name: str, create: bool = True) -> str:
        """Get output directory for a project."""
        pass

    @abstractmethod
    def save_result(self, result: Any, name: str,
                    formats: List[str] = None) -> Dict[str, str]:
        """Save result in specified formats, return paths."""
        pass

    @abstractmethod
    def list_outputs(self, project_name: str) -> List[Dict[str, Any]]:
        """List all outputs for a project."""
        pass

    @abstractmethod
    def cleanup_old(self, project_name: str, keep_latest: int = 10) -> int:
        """Clean up old outputs, return number removed."""
        pass


# =============================================================================
# MONTE CARLO ANALYSIS INTERFACE
# =============================================================================

@dataclass
class UncertaintyParameter:
    """Definition of an uncertain parameter for Monte Carlo analysis."""
    name: str
    nominal: float
    distribution: str  # 'normal', 'uniform', 'triangular'
    std_dev: Optional[float] = None  # For normal
    min_val: Optional[float] = None  # For uniform/triangular
    max_val: Optional[float] = None  # For uniform/triangular
    mode: Optional[float] = None     # For triangular


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo analysis."""
    n_samples: int
    parameters: List[UncertaintyParameter]
    output_names: List[str]
    samples: np.ndarray  # Shape: (n_samples, n_outputs)
    statistics: Dict[str, Dict[str, float]]  # {output: {mean, std, p5, p95, ...}}
    sensitivity: Dict[str, Dict[str, float]]  # {output: {param: correlation}}
    failed_samples: int = 0


class MonteCarloEngine(ABC):
    """Interface for Monte Carlo uncertainty analysis."""

    @abstractmethod
    def add_parameter(self, param: UncertaintyParameter) -> None:
        """Add an uncertain parameter to the analysis."""
        pass

    @abstractmethod
    def run(self, n_samples: int, engine_func: callable,
            output_names: List[str]) -> MonteCarloResult:
        """Run Monte Carlo analysis."""
        pass

    @abstractmethod
    def compute_sensitivity(self, result: MonteCarloResult) -> Dict[str, Dict[str, float]]:
        """Compute sensitivity indices (correlation coefficients)."""
        pass


# =============================================================================
# OPTIMIZATION INTERFACE
# =============================================================================

@dataclass
class OptimizationConstraint:
    """Constraint for optimization problems."""
    name: str
    variable: str
    type: str  # 'min', 'max', 'eq'
    value: float
    tolerance: float = 0.0


@dataclass
class OptimizationResult:
    """Results from optimization."""
    success: bool
    optimal_params: Dict[str, float]
    optimal_value: float
    iterations: int
    constraint_violations: Dict[str, float]
    history: List[Dict[str, float]]


class Optimizer(ABC):
    """Interface for optimization algorithms."""

    @abstractmethod
    def add_variable(self, name: str, min_val: float, max_val: float,
                     initial: Optional[float] = None) -> None:
        """Add a design variable."""
        pass

    @abstractmethod
    def add_constraint(self, constraint: OptimizationConstraint) -> None:
        """Add a constraint."""
        pass

    @abstractmethod
    def set_objective(self, name: str, minimize: bool = True) -> None:
        """Set the objective function output name."""
        pass

    @abstractmethod
    def optimize(self, eval_func: callable, max_iterations: int = 100) -> OptimizationResult:
        """Run optimization."""
        pass
