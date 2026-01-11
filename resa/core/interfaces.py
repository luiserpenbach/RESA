"""
Abstract base classes defining the core interfaces for RESA.

These interfaces enable:
- Swapping implementations (e.g., CEA -> Cantera for combustion)
- Easy testing with mock implementations
- Clear contracts between modules
- Future extensibility without breaking changes
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any, Dict, Optional
from dataclasses import dataclass

# Type variable for generic solver outputs
T = TypeVar('T')


class Solver(ABC, Generic[T]):
    """
    Base interface for all physics solvers.

    Implementations:
    - CEASolver: Rocket combustion using NASA CEA
    - RegenCoolingSolver: Regenerative cooling analysis
    - IsentropicFlowSolver: Compressible flow calculations

    Example:
        class CEASolver(Solver[CombustionResult]):
            def solve(self, pc_bar, mr, eps) -> CombustionResult:
                # Implementation
                pass
    """

    @abstractmethod
    def solve(self, *args, **kwargs) -> T:
        """
        Execute the solver and return typed results.

        Returns:
            Solver-specific result dataclass
        """
        pass

    @abstractmethod
    def validate_inputs(self, *args, **kwargs) -> bool:
        """
        Validate inputs before solving.

        Returns:
            True if inputs are valid

        Raises:
            ValueError: If inputs are invalid with descriptive message
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Return solver metadata for logging/debugging.

        Returns:
            Dictionary with solver name, version, capabilities
        """
        return {
            "name": self.__class__.__name__,
            "module": self.__class__.__module__,
        }


class GeometryGenerator(ABC):
    """
    Base interface for geometry generators.

    Implementations:
    - NozzleGenerator: Bell nozzle contours
    - ChannelGeometryGenerator: Cooling channel dimensions
    - InjectorGenerator: Injector plate geometry

    Example:
        class NozzleGenerator(GeometryGenerator):
            def generate(self, **params) -> NozzleGeometryData:
                pass
    """

    @abstractmethod
    def generate(self, **params) -> 'GeometryData':
        """
        Generate geometry from parameters.

        Returns:
            Geometry dataclass with coordinates and dimensions
        """
        pass

    @abstractmethod
    def validate_parameters(self, **params) -> bool:
        """
        Validate geometry parameters.

        Returns:
            True if parameters are valid
        """
        pass


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

    # Transport properties (optional)
    viscosity: Optional[float] = None     # Pa·s
    conductivity: Optional[float] = None  # W/(m·K)
    cp: Optional[float] = None            # J/(kg·K)


class FluidProvider(ABC):
    """
    Base interface for fluid property providers.

    Implementations:
    - CoolPropProvider: Uses CoolProp library
    - RefpropProvider: Uses REFPROP (if available)
    - IdealGasProvider: Simple ideal gas model

    Example:
        class CoolPropProvider(FluidProvider):
            def __init__(self, fluid_name: str):
                self.fluid = fluid_name

            def get_state(self, p, t) -> FluidState:
                # CoolProp implementation
                pass
    """

    @abstractmethod
    def get_state(self, pressure: float, temperature: float) -> FluidState:
        """
        Get fluid state at given P, T.

        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K

        Returns:
            Complete FluidState at the given conditions
        """
        pass

    @abstractmethod
    def get_state_ph(self, pressure: float, enthalpy: float) -> FluidState:
        """
        Get fluid state at given P, H (useful for isenthalpic processes).

        Args:
            pressure: Pressure in Pa
            enthalpy: Specific enthalpy in J/kg

        Returns:
            FluidState at the given conditions
        """
        pass

    @abstractmethod
    def get_transport_properties(self, state: FluidState) -> Dict[str, float]:
        """
        Get transport properties at a given state.

        Returns:
            Dictionary with 'viscosity', 'conductivity', 'cp', 'prandtl'
        """
        pass

    @property
    @abstractmethod
    def critical_point(self) -> Dict[str, float]:
        """
        Get critical point properties.

        Returns:
            Dictionary with 'T_crit', 'P_crit', 'rho_crit'
        """
        pass


class Plotter(ABC):
    """
    Base interface for all plotters.

    This abstraction enables:
    - Swapping between Plotly/Matplotlib if needed
    - Consistent theming across all plots
    - Easy export to HTML, JSON, or image formats

    All implementations should use Plotly for interactive plots.

    Example:
        class EngineDashboardPlotter(Plotter):
            def create_figure(self, result: EngineDesignResult) -> go.Figure:
                pass
    """

    @abstractmethod
    def create_figure(self, data: Any, **options) -> Any:
        """
        Create a figure from input data.

        Args:
            data: Input data (varies by plotter type)
            options: Plotter-specific options

        Returns:
            Plotly Figure object
        """
        pass

    @abstractmethod
    def to_html(self, figure: Any, **options) -> str:
        """
        Export figure to embeddable HTML string.

        Args:
            figure: Plotly Figure
            options: HTML export options (include_plotlyjs, full_html, etc.)

        Returns:
            HTML string that can be embedded in reports
        """
        pass

    def to_json(self, figure: Any) -> str:
        """
        Export figure to JSON for web/Streamlit embedding.

        Args:
            figure: Plotly Figure

        Returns:
            JSON string representation
        """
        return figure.to_json()

    def show(self, figure: Any) -> None:
        """
        Display figure interactively.

        Works in Jupyter, browser, or Streamlit.
        """
        figure.show()


class ReportGenerator(ABC):
    """
    Base interface for report generation.

    Implementations:
    - HTMLReportGenerator: Rich HTML with embedded Plotly charts
    - PDFReportGenerator: PDF export (using weasyprint)
    - MarkdownReportGenerator: Simple markdown output

    Example:
        class HTMLReportGenerator(ReportGenerator):
            def generate(self, result, output_path=None) -> str:
                # Build HTML from template
                pass
    """

    @abstractmethod
    def generate(
        self,
        result: 'EngineDesignResult',
        output_path: Optional[str] = None,
        **options
    ) -> str:
        """
        Generate report from engine results.

        Args:
            result: Engine design/analysis result
            output_path: If provided, save to this file path
            options: Generator-specific options

        Returns:
            Report content as string (HTML, Markdown, etc.)
        """
        pass

    @abstractmethod
    def add_section(self, title: str, content: str, order: int = 0) -> None:
        """
        Add a custom section to the report.

        Args:
            title: Section heading
            content: HTML/Markdown content
            order: Position in report (lower = earlier)
        """
        pass

    def get_supported_formats(self) -> list:
        """
        Return list of supported output formats.

        Returns:
            List of format strings, e.g., ['html', 'pdf']
        """
        return ['html']
