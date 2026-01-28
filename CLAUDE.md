# CLAUDE.md - AI Assistant Guide for RESA

## Project Overview

**RESA** (Rocket Engine Sizing & Analysis) is a Python toolkit (v2.0.0) for liquid rocket engine preliminary design and analysis. It targets aerospace engineers and covers combustion analysis, regenerative cooling, injector design, throttle analysis, two-phase flow modeling, 3D visualization, and Monte Carlo uncertainty quantification.

**License:** MIT
**Python:** 3.9 - 3.12

## Repository Structure

```
RESA/
├── resa/                          # Main package (v2.0)
│   ├── __init__.py                # Public API exports
│   ├── core/                      # Configuration, results, interfaces, exceptions
│   │   ├── config.py              # EngineConfig dataclass, YAML loading, validation
│   │   ├── engine.py              # Main Engine class orchestrating all solvers
│   │   ├── results.py             # Immutable result dataclasses (frozen=True)
│   │   ├── interfaces.py          # ABCs: Solver, FluidProvider, GeometryGenerator, etc.
│   │   └── exceptions.py          # Exception hierarchy rooted at RESAError
│   ├── physics/                   # Pure physics calculations (no side effects)
│   │   ├── isentropic.py          # Isentropic flow relations, Mach calculations
│   │   ├── heat_transfer.py       # Bartz equation, adiabatic wall temperature
│   │   ├── cooling_n2o.py         # N2O cooling with boiling physics
│   │   └── fluids.py              # CoolProp fluid property provider
│   ├── solvers/                   # Integrated analysis solvers
│   │   ├── combustion.py          # CEASolver wrapping RocketCEA
│   │   └── cooling.py             # Regenerative cooling marching solver
│   ├── geometry/                  # Geometry generators
│   │   ├── nozzle.py              # Rao bell nozzle contour generation
│   │   └── cooling_channels.py    # Cooling channel geometry
│   ├── addons/                    # Specialized design modules
│   │   ├── igniter/               # Torch igniter sizing (CEA, chamber, HEM)
│   │   ├── injector/              # Swirl injector design (LCSC/GCSC)
│   │   ├── contour/               # 3D nozzle contour and STL/DXF export
│   │   └── tank/                  # Tank pressurization and depletion simulation
│   ├── analysis/                  # Statistical and optimization tools
│   │   ├── monte_carlo.py         # Latin Hypercube Sampling uncertainty analysis
│   │   ├── monte_carlo_plots.py   # MC visualization (histograms, tornado, etc.)
│   │   ├── optimization.py        # Single/multi-point design optimization
│   │   └── optimization_plots.py  # Optimization convergence/Pareto plots
│   ├── visualization/             # Plotly-based interactive plots
│   │   ├── engine_plots.py        # 4-panel engine dashboard
│   │   ├── engine_3d.py           # 3D WebGL nozzle viewer
│   │   ├── igniter_plots.py       # Torch igniter visualization
│   │   ├── injector_plots.py      # Swirl injector Cd/spray angle plots
│   │   ├── performance_plots.py   # Isp contours, throttle curves
│   │   └── themes.py              # Centralized PlotTheme system
│   ├── reporting/                 # Report generation
│   │   └── html_report.py         # Professional HTML reports with embedded Plotly
│   ├── ui/                        # Streamlit application
│   │   ├── app.py                 # Main Streamlit entry point
│   │   └── pages/                 # UI pages (design, analysis, injector, igniter, etc.)
│   └── projects/                  # Project and version management
│       ├── project.py             # Project container
│       ├── version_control.py     # Git-based design versioning
│       └── output_manager.py      # Output file organization
├── examples/                      # Example scripts
│   ├── new_architecture_demo.py   # v2.0 architecture showcase
│   └── 2KN_Ethanox_example.py     # Simple 2kN engine design
├── docs/                          # Documentation
├── rocket_engine/                 # Legacy code (pre-v2.0, being replaced)
├── swirl_injector/                # Swirl injector standalone tool
├── torch_igniter_advanced/        # Torch igniter module + tests
├── advanced_contour_design/       # 3D nozzle design tools
├── fluid_lib/                     # Fluid dynamics libraries
├── pyproject.toml                 # Package configuration and tool settings
└── README.md                      # Project documentation
```

## Build & Development

### Installation

```bash
# Development install
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

### Running the Application

```bash
# Streamlit UI
streamlit run resa/ui/app.py

# Or via entry point after install
resa
```

### Running Tests

```bash
# Run all tests
pytest

# Existing test files are in torch_igniter_advanced/
pytest torch_igniter_advanced/

# With coverage
pytest --cov=resa
```

### Code Quality

```bash
# Formatting (line length 100)
black --check .

# Linting
ruff check .
```

**Configured tools (pyproject.toml):**
- **Black**: line-length=100, target py39-py311
- **Ruff**: rules E, F, W, I; ignores E501

## Architecture & Design Patterns

### Key Principles

1. **Separation of Concerns** - Physics modules are pure calculations with no side effects. Solvers orchestrate physics. UI is completely decoupled from business logic.
2. **Dependency Injection** - Solvers accept optional dependencies in `__init__`, enabling mock injection for testing.
3. **Immutable Results** - All result dataclasses use `@dataclass(frozen=True)`.
4. **Interface Contracts** - Abstract base classes in `core/interfaces.py` define contracts (`Solver`, `CombustionSolver`, `CoolingSolver`, `FluidProvider`, `GeometryGenerator`, `Plotter`, `ReportGenerator`).
5. **Custom Exception Hierarchy** - All exceptions inherit from `RESAError` and carry contextual metadata (iterations, residuals, temperatures, pressures).

### Data Flow

```
EngineConfig (YAML/dataclass)
    → Engine.design()
        → CEASolver (combustion)
        → NozzleGenerator (geometry)
        → CoolingChannelGenerator (channels)
        → CoolingSolver (thermal analysis)
    → EngineDesignResult (frozen dataclass)
        → HTMLReportGenerator (report)
        → EngineDashboardPlotter (visualization)
```

### Configuration

Engine designs are configured via `EngineConfig` dataclasses with 50+ parameters. Configs support:
- Programmatic creation via constructor
- YAML loading via `EngineConfig.from_yaml(path)`
- Validation via `config.validate()` returning `ValidationResult` with errors and warnings

Units convention in field names:
- Pressures: `_bar` suffix (e.g., `pc_bar`)
- Temperatures: `_k` suffix (Kelvin)
- Lengths: meters by default, `_mm` suffix for millimeters
- Forces: `_n` suffix (Newtons)

### External Dependencies

- **RocketCEA** (`rocketcea`) - NASA CEA equilibrium combustion calculations
- **CoolProp** - Real fluid thermodynamic properties
- **Plotly** - Interactive visualizations
- **Streamlit** - Web UI framework
- **NumPy/SciPy** - Numerical computing
- **numpy-stl** - STL geometry export

## Code Conventions

### Naming

- **Classes**: `PascalCase` (`EngineConfig`, `CEASolver`, `NozzleGenerator`)
- **Functions/methods**: `snake_case` (`calculate_optimal_expansion`, `_size_throat`)
- **Constants**: `UPPER_SNAKE_CASE` (`G0`, `ETHANOL_LHV`, `PROPELLANT_ALIASES`)
- **Private members**: leading underscore (`_init_solvers`, `_run_combustion`)
- **Parameters**: descriptive `snake_case` with unit suffixes (`pc_bar`, `thrust_n`, `coolant_t_in_k`)

### Type Hints

Extensive use throughout: `Optional`, `Dict`, `List`, `Protocol`, `TYPE_CHECKING`, `Generic[T]`.

### Docstrings

All modules, classes, and public methods have docstrings. Module-level docstrings include purpose, usage examples, and feature lists.

### Logging

Uses Python standard `logging` module with module-level loggers:
```python
logger = logging.getLogger(__name__)
```

### Error Handling

Custom exceptions carry metadata:
```python
raise ConvergenceError("Failed to converge", iterations=100, residual=1e-3)
raise ThermodynamicError("CoolProp failed", fluid="N2O", pressure=5e6, temperature=300)
raise MaterialLimitError("Wall too hot", limit_type="temperature", actual_value=1200, limit_value=900)
```

Non-fatal issues use warning classes (`PerformanceWarning`, `StabilityWarning`, `ThermalWarning`) that are not exceptions.

### Visualization Pattern

All plotters follow a consistent pattern:
- Accept data and `PlotTheme` for styling
- Provide `create_figure()` returning a Plotly figure
- Provide `to_html()` for report embedding
- Use subplots for multi-panel dashboards

## Key Public API

```python
from resa import Engine, EngineConfig

config = EngineConfig(
    engine_name="Phoenix-1",
    fuel="Ethanol90",
    oxidizer="N2O",
    thrust_n=2200,
    pc_bar=25,
    mr=4.0,
)

engine = Engine(config)
result = engine.design()
result.to_html("report.html")
```

Important public exports (from `resa/__init__.py`):
- `Engine`, `EngineConfig`, `EngineDesignResult`, `CombustionResult`
- `Solver`, `FluidProvider`, `GeometryGenerator` (interfaces)
- `RESAError`, `ConfigurationError`, `PhysicsError` (exceptions)
- `PlotTheme`, `EngineeringTheme`, `DarkTheme` (visualization themes)
- `EngineDashboardPlotter`, `CrossSectionPlotter`, `Engine3DViewer`
- `HTMLReportGenerator`
- `MonteCarloAnalysis`, `ThrottleOptimizer`

## Guidelines for Making Changes

1. **Physics modules** (`resa/physics/`) must remain pure functions with no side effects or state. They should not import from `solvers/` or `ui/`.
2. **New solvers** should implement the appropriate ABC from `core/interfaces.py`.
3. **Result types** must be frozen dataclasses (`@dataclass(frozen=True)`).
4. **New addons** go in `resa/addons/<module_name>/` and should implement `AnalysisModule` for UI integration.
5. **Visualization** code goes in `resa/visualization/` using Plotly and the `PlotTheme` system.
6. **UI pages** go in `resa/ui/pages/` as Streamlit page modules.
7. **Keep the dependency direction**: `core` depends on nothing; `physics` depends on `core`; `solvers` depends on `core` + `physics`; `visualization`/`ui`/`reporting` depend on everything above but never the reverse.
8. **Legacy directories** (`rocket_engine/`, `fluid_lib/`, `swirl_injector/`, `torch_igniter_advanced/`, `advanced_contour_design/`) contain older code being migrated into the `resa/` package. New development should go in `resa/`.
9. **Test files** currently live in `torch_igniter_advanced/`. New tests should use `pytest` conventions.
10. **Format code** with Black (line-length 100) and lint with Ruff before committing.
