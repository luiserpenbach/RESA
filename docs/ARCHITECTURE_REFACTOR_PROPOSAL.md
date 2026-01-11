# RESA Architecture Refactoring Proposal

## Executive Summary

This document proposes a comprehensive refactoring of the RESA (Rocket Engine Sizing & Analysis) codebase to:
1. Improve modularity and extensibility for future modules
2. Standardize on **Plotly** for all visualizations
3. Implement an **HTML report generation** system
4. Prepare the architecture for future **Streamlit UI** integration
5. Follow professional Python package conventions

---

## Current Architecture Analysis

### Strengths
- Well-structured dataclasses (`EngineConfig`, `EngineDesignResult`)
- Clear separation between physics, geometry, and analysis
- YAML configuration support
- Existing Streamlit prototype

### Issues Identified

| Issue | Location | Impact |
|-------|----------|--------|
| **Mixed visualization libraries** | `visualization.py`, `engine.py` | Matplotlib scattered throughout, hard to swap |
| **Tight coupling** | `engine.py:734-856` | Plotting logic mixed with physics calculations |
| **No abstract interfaces** | All modules | Can't swap implementations (e.g., different solvers) |
| **Print statements for logging** | `engine.py:169,175,216` | No proper logging framework |
| **Hardcoded parameters** | `heat_transfer.py` | Magic numbers for viscosity, Cp, Prandtl |
| **Missing type hints** | Various | Reduces IDE support and documentation |
| **No dependency injection** | `LiquidEngine.__init__` | Hard to test or extend |
| **Report generation separate** | `reporting/` | Not integrated with main engine output |

---

## Proposed New Architecture

### Directory Structure

```
RESA/
├── pyproject.toml              # Modern Python packaging
├── README.md
├── docs/
│   └── ARCHITECTURE.md
│
├── resa/                       # Main package (renamed from rocket_engine)
│   ├── __init__.py
│   ├── config/                 # Configuration management
│   │   ├── __init__.py
│   │   ├── engine_config.py    # EngineConfig dataclass
│   │   ├── defaults.py         # Default values & constants
│   │   └── loader.py           # YAML/JSON config loading
│   │
│   ├── core/                   # Core domain models
│   │   ├── __init__.py
│   │   ├── engine.py           # LiquidEngine (no plotting)
│   │   ├── results.py          # EngineDesignResult, AnalysisResult
│   │   └── interfaces.py       # Abstract base classes
│   │
│   ├── solvers/                # Physics solvers (pluggable)
│   │   ├── __init__.py
│   │   ├── base.py             # AbstractSolver interface
│   │   ├── combustion/
│   │   │   ├── __init__.py
│   │   │   ├── cea_solver.py   # RocketCEA implementation
│   │   │   └── cantera_solver.py  # Future: Cantera option
│   │   ├── cooling/
│   │   │   ├── __init__.py
│   │   │   └── regen_solver.py
│   │   ├── flow/
│   │   │   ├── __init__.py
│   │   │   └── isentropic.py
│   │   └── thermal/
│   │       ├── __init__.py
│   │       └── bartz.py
│   │
│   ├── geometry/               # Geometry generators
│   │   ├── __init__.py
│   │   ├── base.py             # AbstractGeometryGenerator
│   │   ├── nozzle.py
│   │   ├── cooling_channels.py
│   │   └── injector.py
│   │
│   ├── fluids/                 # Fluid property management
│   │   ├── __init__.py
│   │   ├── base.py             # AbstractFluidProvider
│   │   ├── coolprop_provider.py
│   │   └── propellants.py
│   │
│   ├── visualization/          # ALL PLOTLY-BASED
│   │   ├── __init__.py
│   │   ├── base.py             # AbstractPlotter interface
│   │   ├── engine_plots.py     # Engine contour, thermal plots
│   │   ├── cooling_plots.py    # Cooling system visualizations
│   │   ├── performance_plots.py # Isp contours, throttle curves
│   │   ├── cross_section.py    # 2D channel cross-sections
│   │   ├── engine_3d.py        # 3D engine visualization
│   │   └── themes.py           # Consistent color schemes
│   │
│   ├── reporting/              # HTML report generation
│   │   ├── __init__.py
│   │   ├── base.py             # AbstractReportGenerator
│   │   ├── html_report.py      # Main HTML report builder
│   │   ├── pdf_export.py       # Optional PDF conversion
│   │   └── templates/
│   │       ├── design_report.html
│   │       ├── analysis_report.html
│   │       └── components/
│   │           ├── header.html
│   │           ├── metrics_card.html
│   │           └── plot_section.html
│   │
│   ├── export/                 # CAD/Data export
│   │   ├── __init__.py
│   │   ├── dxf.py
│   │   ├── csv.py
│   │   └── step.py             # Future: STEP export
│   │
│   ├── analysis/               # Analysis utilities
│   │   ├── __init__.py
│   │   ├── throttle.py
│   │   ├── sensitivity.py
│   │   └── transient.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── units.py
│       ├── logging.py          # Proper logging setup
│       └── validation.py       # Input validation
│
├── ui/                         # Streamlit UI (separate package)
│   ├── __init__.py
│   ├── app.py                  # Main Streamlit entry
│   ├── pages/
│   │   ├── design.py
│   │   ├── analysis.py
│   │   └── reports.py
│   └── components/
│       ├── sidebar.py
│       └── charts.py           # Streamlit-wrapped Plotly
│
├── examples/
│   ├── basic_design.py
│   ├── throttle_analysis.py
│   └── custom_report.py
│
└── tests/
    ├── conftest.py
    ├── test_engine.py
    ├── test_solvers/
    └── test_visualization/
```

---

## Key Design Patterns

### 1. Abstract Base Classes (Interfaces)

```python
# resa/core/interfaces.py
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, Generic
from dataclasses import dataclass

T = TypeVar('T')

class Solver(ABC, Generic[T]):
    """Base interface for all physics solvers."""

    @abstractmethod
    def solve(self, *args, **kwargs) -> T:
        """Execute the solver and return results."""
        pass

    @abstractmethod
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate inputs before solving."""
        pass


class GeometryGenerator(ABC):
    """Base interface for geometry generators."""

    @abstractmethod
    def generate(self, **params) -> 'GeometryData':
        pass


class FluidProvider(ABC):
    """Base interface for fluid property providers."""

    @abstractmethod
    def get_state(self, pressure: float, temperature: float) -> 'FluidState':
        pass

    @abstractmethod
    def get_transport_properties(self, state: 'FluidState') -> dict:
        pass


class Plotter(ABC):
    """Base interface for all plotters - enables swapping Plotly/Matplotlib."""

    @abstractmethod
    def create_figure(self, data: dict, **options) -> 'Figure':
        pass

    @abstractmethod
    def to_html(self, figure: 'Figure') -> str:
        """Export figure to embeddable HTML."""
        pass


class ReportGenerator(ABC):
    """Base interface for report generation."""

    @abstractmethod
    def generate(self, result: 'EngineDesignResult', **options) -> str:
        """Generate report and return path or HTML string."""
        pass
```

### 2. Dependency Injection Pattern

```python
# resa/core/engine.py
from dataclasses import dataclass, field
from typing import Optional
import logging

from resa.core.interfaces import Solver, GeometryGenerator, FluidProvider
from resa.solvers.combustion.cea_solver import CEASolver
from resa.solvers.cooling.regen_solver import RegenCoolingSolver
from resa.geometry.nozzle import NozzleGenerator
from resa.fluids.coolprop_provider import CoolPropProvider

logger = logging.getLogger(__name__)


@dataclass
class EngineComponents:
    """Dependency container for engine solvers and generators."""
    combustion_solver: Solver
    cooling_solver: Solver
    nozzle_generator: GeometryGenerator
    fluid_provider: FluidProvider


class LiquidEngine:
    """
    Main rocket engine analysis class.

    Uses dependency injection for all solvers, enabling:
    - Easy testing with mock solvers
    - Swapping implementations (CEA -> Cantera)
    - Future extensibility
    """

    def __init__(
        self,
        config: 'EngineConfig',
        components: Optional[EngineComponents] = None
    ):
        self.config = config
        self._logger = logging.getLogger(f"{__name__}.{config.engine_name}")

        # Use provided components or create defaults
        if components is None:
            components = self._create_default_components()

        self._combustion = components.combustion_solver
        self._cooling = components.cooling_solver
        self._nozzle_gen = components.nozzle_generator
        self._fluid = components.fluid_provider

        self._logger.info(f"Initialized engine: {config.engine_name}")

    def _create_default_components(self) -> EngineComponents:
        """Factory for default solver implementations."""
        return EngineComponents(
            combustion_solver=CEASolver(
                self.config.fuel,
                self.config.oxidizer
            ),
            cooling_solver=RegenCoolingSolver(),
            nozzle_generator=NozzleGenerator(),
            fluid_provider=CoolPropProvider(self.config.coolant_name)
        )

    def design(self) -> 'EngineDesignResult':
        """
        Execute design point calculation.

        Returns pure data - no side effects, no plotting.
        """
        self._logger.info(
            f"Running design: Thrust={self.config.thrust_n}N, "
            f"Pc={self.config.pc_bar}bar"
        )

        # 1. Combustion analysis
        combustion = self._combustion.solve(
            pc_bar=self.config.pc_bar,
            mr=self.config.mr,
            expansion_ratio=self._get_expansion_ratio()
        )

        # 2. Geometry generation
        geometry = self._nozzle_gen.generate(
            throat_radius=self._calculate_throat_radius(combustion),
            expansion_ratio=combustion.expansion_ratio,
            L_star=self.config.L_star,
            # ... other params
        )

        # 3. Physics analysis
        physics = self._run_physics_loop(combustion, geometry)

        # 4. Cooling analysis
        cooling = self._cooling.solve(
            geometry=geometry,
            physics=physics,
            config=self.config
        )

        return EngineDesignResult(
            config=self.config,
            combustion=combustion,
            geometry=geometry,
            physics=physics,
            cooling=cooling
        )
```

---

## Plotly Visualization Layer

### Design Philosophy
- **Separation of concerns**: Engine calculations return data, visualization is separate
- **Consistent theming**: Single source of truth for colors, fonts, layouts
- **HTML-first**: All plots exportable to standalone HTML
- **Interactive**: Leverage Plotly's hover, zoom, and selection features

### Core Visualization Classes

```python
# resa/visualization/base.py
from abc import ABC, abstractmethod
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any, List
import json


class PlotlyPlotter(ABC):
    """Base class for all Plotly-based visualizations."""

    def __init__(self, theme: Optional['PlotTheme'] = None):
        self.theme = theme or DefaultTheme()

    @abstractmethod
    def create_figure(self, data: Any) -> go.Figure:
        """Create a Plotly figure from input data."""
        pass

    def to_html(
        self,
        fig: go.Figure,
        include_plotlyjs: bool = True,
        full_html: bool = False
    ) -> str:
        """Export figure to embeddable HTML."""
        return fig.to_html(
            include_plotlyjs=include_plotlyjs,
            full_html=full_html
        )

    def to_json(self, fig: go.Figure) -> str:
        """Export figure to JSON for Streamlit/web embedding."""
        return fig.to_json()

    def show(self, fig: go.Figure):
        """Display figure (works in Jupyter, browser, or Streamlit)."""
        fig.show()


# resa/visualization/themes.py
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PlotTheme:
    """Centralized theme configuration for all plots."""

    # Colors
    primary: str = "#1f77b4"
    secondary: str = "#ff7f0e"
    accent: str = "#2ca02c"
    danger: str = "#d62728"

    # Material colors
    copper: str = "#B87333"
    coolant: str = "#4db6ac"
    gas: str = "#ffccbc"
    steel: str = "#A9A9A9"

    # Line styles
    line_width: int = 2

    # Fonts
    font_family: str = "Arial, sans-serif"
    title_size: int = 16
    axis_size: int = 12

    # Layout
    margin: Dict[str, int] = None

    def __post_init__(self):
        if self.margin is None:
            self.margin = {"l": 60, "r": 60, "t": 80, "b": 60}

    def apply_to_figure(self, fig: 'go.Figure') -> 'go.Figure':
        """Apply theme to a Plotly figure."""
        fig.update_layout(
            font=dict(family=self.font_family),
            title_font_size=self.title_size,
            margin=self.margin,
            template="plotly_white"
        )
        return fig


class EngineeringTheme(PlotTheme):
    """Professional engineering-focused theme."""

    primary: str = "#2c3e50"
    secondary: str = "#e74c3c"
    accent: str = "#27ae60"

    # Specific to engine plots
    temperature_colorscale: List = None
    pressure_colorscale: List = None

    def __post_init__(self):
        super().__post_init__()
        self.temperature_colorscale = [
            [0, "#3498db"],    # Cool blue
            [0.5, "#f1c40f"],  # Yellow
            [1, "#e74c3c"]     # Hot red
        ]
        self.pressure_colorscale = [
            [0, "#ecf0f1"],
            [1, "#2c3e50"]
        ]
```

### Engine Visualization Module

```python
# resa/visualization/engine_plots.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional

from resa.visualization.base import PlotlyPlotter
from resa.visualization.themes import EngineeringTheme
from resa.core.results import EngineDesignResult


class EngineDashboardPlotter(PlotlyPlotter):
    """
    Creates the main 4-panel engine analysis dashboard.

    Panels:
    1. Geometry & Thermal State
    2. Gas Dynamics & Heat Flux
    3. Coolant Pressure Evolution
    4. Coolant Flow Properties
    """

    def __init__(self, theme: Optional[EngineeringTheme] = None):
        super().__init__(theme or EngineeringTheme())

    def create_figure(self, result: EngineDesignResult) -> go.Figure:
        """Create the full engine dashboard."""

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "Chamber Geometry & Thermal State",
                "Gas Dynamics & Heat Flux Profile",
                "Coolant Pressure Drop",
                "Coolant Hydraulics"
            ),
            specs=[
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"secondary_y": False}],
                [{"secondary_y": True}]
            ]
        )

        # Extract data
        x_mm = result.geometry.x_full
        y_mm = result.geometry.y_full
        cooling = result.cooling_data

        # --- Panel 1: Geometry & Temperature ---
        fig.add_trace(
            go.Scatter(
                x=x_mm, y=y_mm,
                name="Chamber Wall",
                line=dict(color="black", width=3),
                fill='tozeroy',
                fillcolor='rgba(184, 115, 51, 0.1)'
            ),
            row=1, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=x_mm, y=cooling['T_wall_hot'],
                name="Hot Wall Temp",
                line=dict(color=self.theme.danger, width=2)
            ),
            row=1, col=1, secondary_y=True
        )

        fig.add_trace(
            go.Scatter(
                x=x_mm, y=cooling['T_coolant'],
                name="Coolant Temp",
                line=dict(color=self.theme.primary, width=2, dash='dash')
            ),
            row=1, col=1, secondary_y=True
        )

        # --- Panel 2: Mach & Heat Flux ---
        fig.add_trace(
            go.Scatter(
                x=x_mm, y=result.mach_numbers,
                name="Mach Number",
                line=dict(color=self.theme.accent, width=2)
            ),
            row=2, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=x_mm, y=cooling['q_flux'] / 1e6,
                name="Heat Flux",
                line=dict(color="#9b59b6", width=2)
            ),
            row=2, col=1, secondary_y=True
        )

        # --- Panel 3: Pressure ---
        fig.add_trace(
            go.Scatter(
                x=x_mm, y=cooling['P_coolant'] / 1e5,
                name="Coolant Pressure",
                line=dict(color="#17a2b8", width=2.5),
                fill='tozeroy',
                fillcolor='rgba(23, 162, 184, 0.1)'
            ),
            row=3, col=1
        )

        # --- Panel 4: Velocity & Density ---
        fig.add_trace(
            go.Scatter(
                x=x_mm, y=cooling['velocity'],
                name="Velocity",
                line=dict(color="black", width=2)
            ),
            row=4, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=x_mm, y=cooling['density'],
                name="Density",
                line=dict(color=self.theme.primary, width=2, dash='dot')
            ),
            row=4, col=1, secondary_y=True
        )

        # Update axes labels
        fig.update_yaxes(title_text="Radius [mm]", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Temperature [K]", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Mach [-]", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Heat Flux [MW/m²]", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Pressure [bar]", row=3, col=1)
        fig.update_yaxes(title_text="Velocity [m/s]", row=4, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Density [kg/m³]", row=4, col=1, secondary_y=True)
        fig.update_xaxes(title_text="Axial Position [mm]", row=4, col=1)

        # Apply theme
        fig.update_layout(
            height=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            title=dict(
                text=f"Engine Analysis: {result.config.engine_name}",
                x=0.5,
                font=dict(size=20)
            )
        )

        self.theme.apply_to_figure(fig)

        return fig


class CrossSectionPlotter(PlotlyPlotter):
    """Plots cooling channel cross-sections using Plotly shapes."""

    def create_figure(
        self,
        channel_geometry: 'CoolingChannelGeometry',
        station_idx: int,
        sector_angle: float = 90.0
    ) -> go.Figure:
        """Create radial cross-section view."""

        # Extract dimensions at station
        R_inner = channel_geometry.radius_contour[station_idx] * 1000
        t_wall = channel_geometry.wall_thickness[station_idx] * 1000
        w_ch = channel_geometry.channel_width[station_idx] * 1000
        h_ch = channel_geometry.channel_height[station_idx] * 1000
        w_rib = channel_geometry.rib_width[station_idx] * 1000
        n_channels = channel_geometry.number_of_channels

        R_channel_base = R_inner + t_wall
        R_channel_top = R_channel_base + h_ch
        R_outer = R_channel_top + 1.0  # 1mm closeout

        fig = go.Figure()

        # Create shapes for the annular sections
        theta = np.linspace(0, np.radians(sector_angle), 100)

        # Inner liner (copper)
        r_inner_x = R_inner * np.cos(theta)
        r_inner_y = R_inner * np.sin(theta)
        r_base_x = R_channel_base * np.cos(theta)
        r_base_y = R_channel_base * np.sin(theta)

        fig.add_trace(go.Scatter(
            x=np.concatenate([r_inner_x, r_base_x[::-1]]),
            y=np.concatenate([r_inner_y, r_base_y[::-1]]),
            fill='toself',
            fillcolor=self.theme.copper,
            line=dict(color='black', width=1),
            name='Liner (Cu)'
        ))

        # Draw channels and ribs
        theta_pitch = 2 * np.pi / n_channels
        theta_ch = theta_pitch * (w_ch / (w_ch + w_rib))

        channels_to_draw = int(np.ceil(np.radians(sector_angle) / theta_pitch))

        for i in range(channels_to_draw):
            start_angle = i * theta_pitch

            # Channel (coolant)
            ch_theta = np.linspace(start_angle, start_angle + theta_ch, 20)
            ch_inner_x = R_channel_base * np.cos(ch_theta)
            ch_inner_y = R_channel_base * np.sin(ch_theta)
            ch_outer_x = R_channel_top * np.cos(ch_theta)
            ch_outer_y = R_channel_top * np.sin(ch_theta)

            fig.add_trace(go.Scatter(
                x=np.concatenate([ch_inner_x, ch_outer_x[::-1]]),
                y=np.concatenate([ch_inner_y, ch_outer_y[::-1]]),
                fill='toself',
                fillcolor=self.theme.coolant,
                line=dict(color='black', width=0.5),
                name='Coolant' if i == 0 else None,
                showlegend=(i == 0)
            ))

            # Rib
            rib_theta = np.linspace(start_angle + theta_ch, start_angle + theta_pitch, 20)
            rib_inner_x = R_channel_base * np.cos(rib_theta)
            rib_inner_y = R_channel_base * np.sin(rib_theta)
            rib_outer_x = R_channel_top * np.cos(rib_theta)
            rib_outer_y = R_channel_top * np.sin(rib_theta)

            fig.add_trace(go.Scatter(
                x=np.concatenate([rib_inner_x, rib_outer_x[::-1]]),
                y=np.concatenate([rib_inner_y, rib_outer_y[::-1]]),
                fill='toself',
                fillcolor=self.theme.copper,
                line=dict(color='black', width=0.5),
                showlegend=False
            ))

        # Closeout jacket
        r_top_x = R_channel_top * np.cos(theta)
        r_top_y = R_channel_top * np.sin(theta)
        r_outer_x = R_outer * np.cos(theta)
        r_outer_y = R_outer * np.sin(theta)

        fig.add_trace(go.Scatter(
            x=np.concatenate([r_top_x, r_outer_x[::-1]]),
            y=np.concatenate([r_top_y, r_outer_y[::-1]]),
            fill='toself',
            fillcolor=self.theme.steel,
            line=dict(color='black', width=1),
            name='Closeout'
        ))

        # Center annotation
        fig.add_annotation(
            x=0, y=0,
            text=f"N={n_channels}<br>X={channel_geometry.x_contour[station_idx]*1000:.1f}mm",
            showarrow=False,
            font=dict(size=12, color='black')
        )

        fig.update_layout(
            title=f"Radial Cross-Section at Station {station_idx}",
            xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            width=700,
            height=700
        )

        return fig
```

---

## HTML Report Generation System

### Report Architecture

```python
# resa/reporting/html_report.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape
import json
from datetime import datetime

from resa.core.results import EngineDesignResult
from resa.visualization.engine_plots import EngineDashboardPlotter, CrossSectionPlotter
from resa.visualization.themes import EngineeringTheme


@dataclass
class ReportSection:
    """Represents a section in the report."""
    title: str
    content: str  # HTML content
    order: int = 0


class HTMLReportGenerator:
    """
    Generates comprehensive HTML reports from engine analysis results.

    Features:
    - Embedded interactive Plotly charts
    - Responsive design for web/print
    - Customizable sections
    - Export to standalone HTML or PDF
    """

    TEMPLATE_DIR = Path(__file__).parent / "templates"

    def __init__(
        self,
        theme: Optional[EngineeringTheme] = None,
        template_name: str = "design_report.html"
    ):
        self.theme = theme or EngineeringTheme()
        self.template_name = template_name

        # Setup Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.TEMPLATE_DIR)),
            autoescape=select_autoescape(['html', 'xml'])
        )

        # Initialize plotters
        self.dashboard_plotter = EngineDashboardPlotter(self.theme)
        self.cross_section_plotter = CrossSectionPlotter(self.theme)

    def generate(
        self,
        result: EngineDesignResult,
        output_path: Optional[str] = None,
        include_3d: bool = True,
        additional_sections: Optional[List[ReportSection]] = None
    ) -> str:
        """
        Generate HTML report from engine results.

        Args:
            result: Engine design/analysis result
            output_path: If provided, save to file
            include_3d: Whether to include 3D visualizations
            additional_sections: Custom sections to add

        Returns:
            HTML string of the complete report
        """

        # Generate all plots
        plots = self._generate_plots(result, include_3d)

        # Build context for template
        context = {
            'title': f"Engine Design Report: {result.config.engine_name}",
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'engine_name': result.config.engine_name,

            # Configuration
            'config': self._config_to_dict(result.config),

            # Performance metrics
            'performance': {
                'isp_vac': f"{result.isp_vac:.1f}",
                'isp_sea': f"{result.isp_sea:.1f}",
                'thrust_vac': f"{result.thrust_vac:.0f}",
                'thrust_sea': f"{result.thrust_sea:.0f}",
                'massflow': f"{result.massflow_total:.4f}",
                'expansion_ratio': f"{result.expansion_ratio:.2f}",
            },

            # Geometry
            'geometry': {
                'throat_diameter': f"{result.dt_mm:.2f}",
                'exit_diameter': f"{result.de_mm:.2f}",
                'length': f"{result.length_mm:.1f}",
            },

            # Cooling summary
            'cooling': self._cooling_summary(result),

            # Embedded plots (as HTML strings)
            'plots': plots,

            # Additional sections
            'extra_sections': additional_sections or []
        }

        # Render template
        template = self.env.get_template(self.template_name)
        html_content = template.render(**context)

        # Save if path provided
        if output_path:
            Path(output_path).write_text(html_content)

        return html_content

    def _generate_plots(
        self,
        result: EngineDesignResult,
        include_3d: bool
    ) -> Dict[str, str]:
        """Generate all plots and return as embeddable HTML."""

        plots = {}

        # Main dashboard
        dashboard_fig = self.dashboard_plotter.create_figure(result)
        plots['dashboard'] = dashboard_fig.to_html(
            include_plotlyjs='cdn',  # Use CDN for smaller file
            full_html=False
        )

        # Cross-section at throat
        throat_idx = result.geometry.y_full.argmin()
        cross_section_fig = self.cross_section_plotter.create_figure(
            result.channel_geometry,
            station_idx=throat_idx,
            sector_angle=90
        )
        plots['cross_section'] = cross_section_fig.to_html(
            include_plotlyjs=False,
            full_html=False
        )

        # TODO: Add 3D plots when Plotly 3D is implemented

        return plots

    def _config_to_dict(self, config: 'EngineConfig') -> Dict[str, Any]:
        """Convert config to display-friendly dictionary."""
        return {
            'Fuel': config.fuel,
            'Oxidizer': config.oxidizer,
            'Chamber Pressure': f"{config.pc_bar} bar",
            'Mixture Ratio': f"{config.mr:.2f}",
            'Target Thrust': f"{config.thrust_n} N",
            'L*': f"{config.L_star} mm",
            'Coolant': config.coolant_name.split('::')[-1],
            'Cooling Mode': config.cooling_mode,
        }

    def _cooling_summary(self, result: EngineDesignResult) -> Dict[str, str]:
        """Extract cooling system summary metrics."""
        cooling = result.cooling_data
        return {
            'max_wall_temp': f"{cooling['T_wall_hot'].max():.0f} K",
            'max_coolant_temp': f"{cooling['T_coolant'].max():.0f} K",
            'max_heat_flux': f"{cooling['q_flux'].max() / 1e6:.2f} MW/m²",
            'pressure_drop': f"{(cooling['P_coolant'].max() - cooling['P_coolant'].min()) / 1e5:.2f} bar",
            'min_density': f"{cooling['density'].min():.1f} kg/m³",
        }
```

### HTML Template

```html
<!-- resa/reporting/templates/design_report.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {
            --primary: #2c3e50;
            --secondary: #3498db;
            --accent: #e74c3c;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
        }

        * { box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: var(--bg);
            margin: 0;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            background: var(--primary);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }

        header h1 { margin: 0 0 10px 0; }
        header .meta { opacity: 0.8; font-size: 0.9em; }

        .card {
            background: var(--card-bg);
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 25px;
            margin-bottom: 25px;
        }

        .card h2 {
            color: var(--primary);
            border-bottom: 2px solid var(--secondary);
            padding-bottom: 10px;
            margin-top: 0;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }

        .metric {
            text-align: center;
            padding: 20px;
            background: var(--bg);
            border-radius: 8px;
        }

        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: var(--primary);
        }

        .metric-label {
            color: #666;
            font-size: 0.9em;
        }

        .plot-container {
            width: 100%;
            min-height: 400px;
        }

        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }

        @media (max-width: 900px) {
            .two-column { grid-template-columns: 1fr; }
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        th { background: var(--bg); }

        @media print {
            .card { break-inside: avoid; }
            header { background: var(--primary) !important; -webkit-print-color-adjust: exact; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ engine_name }}</h1>
            <p class="meta">Generated: {{ generated_at }}</p>
        </header>

        <!-- Performance Summary -->
        <div class="card">
            <h2>Performance Summary</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{{ performance.isp_vac }}</div>
                    <div class="metric-label">Vacuum Isp [s]</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ performance.thrust_vac }}</div>
                    <div class="metric-label">Vacuum Thrust [N]</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ performance.massflow }}</div>
                    <div class="metric-label">Mass Flow [kg/s]</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ performance.expansion_ratio }}</div>
                    <div class="metric-label">Expansion Ratio</div>
                </div>
            </div>
        </div>

        <!-- Configuration & Geometry -->
        <div class="two-column">
            <div class="card">
                <h2>Configuration</h2>
                <table>
                    {% for key, value in config.items() %}
                    <tr>
                        <th>{{ key }}</th>
                        <td>{{ value }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>

            <div class="card">
                <h2>Geometry</h2>
                <table>
                    <tr><th>Throat Diameter</th><td>{{ geometry.throat_diameter }} mm</td></tr>
                    <tr><th>Exit Diameter</th><td>{{ geometry.exit_diameter }} mm</td></tr>
                    <tr><th>Chamber Length</th><td>{{ geometry.length }} mm</td></tr>
                </table>

                <h3>Cooling Summary</h3>
                <table>
                    {% for key, value in cooling.items() %}
                    <tr>
                        <th>{{ key | replace('_', ' ') | title }}</th>
                        <td>{{ value }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </div>

        <!-- Main Dashboard Plot -->
        <div class="card">
            <h2>Thermal & Flow Analysis</h2>
            <div class="plot-container">
                {{ plots.dashboard | safe }}
            </div>
        </div>

        <!-- Cross Section -->
        <div class="card">
            <h2>Channel Cross-Section (Throat)</h2>
            <div class="plot-container" style="max-width: 700px; margin: 0 auto;">
                {{ plots.cross_section | safe }}
            </div>
        </div>

        <!-- Additional Sections -->
        {% for section in extra_sections %}
        <div class="card">
            <h2>{{ section.title }}</h2>
            {{ section.content | safe }}
        </div>
        {% endfor %}
    </div>
</body>
</html>
```

---

## Streamlit UI Preparation

The new architecture makes Streamlit integration seamless:

```python
# ui/app.py
import streamlit as st
from resa import LiquidEngine, EngineConfig
from resa.visualization.engine_plots import EngineDashboardPlotter
from resa.reporting.html_report import HTMLReportGenerator

st.set_page_config(page_title="RESA", layout="wide")

# Sidebar configuration...
config = EngineConfig(...)
engine = LiquidEngine(config)

if st.button("Run Design"):
    result = engine.design()

    # Display Plotly chart directly in Streamlit
    plotter = EngineDashboardPlotter()
    fig = plotter.create_figure(result)
    st.plotly_chart(fig, use_container_width=True)

    # Generate downloadable HTML report
    reporter = HTMLReportGenerator()
    html_report = reporter.generate(result)
    st.download_button(
        "Download HTML Report",
        html_report,
        file_name=f"{config.engine_name}_report.html",
        mime="text/html"
    )
```

---

## Migration Plan

### Phase 1: Core Restructuring
1. Create new package structure under `resa/`
2. Implement abstract interfaces
3. Migrate existing code to new modules
4. Add proper logging

### Phase 2: Visualization Layer
1. Implement Plotly-based plotters
2. Create theming system
3. Migrate all matplotlib plots to Plotly
4. Test interactive features

### Phase 3: Reporting System
1. Implement HTML report generator
2. Create report templates
3. Add PDF export (optional)
4. Integrate with main engine class

### Phase 4: UI Integration
1. Refactor Streamlit app to use new visualization
2. Add report generation to UI
3. Implement caching for performance

---

## Dependencies Update

```toml
# pyproject.toml
[project]
name = "resa"
version = "2.0.0"
requires-python = ">=3.9"

dependencies = [
    "numpy>=1.21",
    "scipy>=1.7",
    "pandas>=1.3",
    "plotly>=5.15",
    "CoolProp>=6.4",
    "rocketcea>=1.1",
    "pyyaml>=6.0",
    "jinja2>=3.0",
]

[project.optional-dependencies]
ui = ["streamlit>=1.25"]
pdf = ["weasyprint>=59"]
cad = ["ezdxf>=1.0"]
dev = ["pytest", "pytest-cov", "mypy", "ruff"]
```

---

## Summary

This refactoring provides:

| Feature | Current | Proposed |
|---------|---------|----------|
| Visualization | Matplotlib (scattered) | Plotly (centralized, interactive) |
| Reports | Basic PDF | Rich HTML with embedded charts |
| Architecture | Tightly coupled | Interface-based, pluggable |
| Extensibility | Hard to add solvers | Easy via dependency injection |
| Testing | Difficult | Easy with mock injections |
| Streamlit | Prototype | Production-ready integration |
| Logging | Print statements | Proper logging framework |
