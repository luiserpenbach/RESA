# CLAUDE.md - AI Assistant Guide for RESA

## Project Overview

**RESA** (Rocket Engine Sizing & Analysis) is a Python toolkit (v2.0.0) for liquid rocket engine preliminary design and analysis. It targets aerospace engineers and covers combustion analysis, regenerative cooling, injector design, throttle analysis, two-phase flow modeling, 3D visualization, and Monte Carlo uncertainty quantification.

**License:** MIT
**Python:** 3.9 - 3.12

## Repository Structure

```
RESA/
├── resa/                          # Main package (v2.0) — 72 Python files, ~25k lines
│   ├── __init__.py                # Public API exports
│   ├── core/                      # Configuration, results, interfaces, exceptions
│   │   ├── config.py              # EngineConfig dataclass, YAML loading, validation
│   │   ├── engine.py              # Main Engine class orchestrating all solvers
│   │   ├── results.py             # Immutable result dataclasses (frozen=True)
│   │   ├── interfaces.py          # ABCs: Solver, FluidProvider, GeometryGenerator, etc.
│   │   ├── exceptions.py          # Exception hierarchy rooted at RESAError
│   │   ├── materials.py           # Material property definitions
│   │   ├── module_configs.py      # Addon module configuration helpers
│   │   └── session.py             # Session state management
│   ├── physics/                   # Pure physics calculations (no side effects)
│   │   ├── isentropic.py          # Isentropic flow relations, Mach calculations
│   │   ├── heat_transfer.py       # Bartz equation, adiabatic wall temperature
│   │   ├── cooling_n2o.py         # N2O cooling with two-phase boiling physics (~1.5k lines)
│   │   ├── fluids.py              # CoolProp fluid property provider
│   │   ├── performance.py         # Engine performance calculations
│   │   ├── structural.py          # Structural analysis (chamber/nozzle wall)
│   │   └── feed_system.py         # Feed system hydraulics
│   ├── solvers/                   # Integrated analysis solvers
│   │   ├── combustion.py          # CEASolver wrapping RocketCEA
│   │   ├── cooling.py             # Regenerative cooling marching solver
│   │   ├── performance.py         # Performance solver
│   │   ├── structural.py          # Structural solver
│   │   └── feed_system.py         # Feed system solver
│   ├── geometry/                  # Geometry generators
│   │   ├── nozzle.py              # Rao bell nozzle contour generation
│   │   └── cooling_channels.py    # Cooling channel geometry
│   ├── addons/                    # Specialized design modules
│   │   ├── igniter/               # Torch igniter sizing (CEA, chamber, HEM two-phase)
│   │   ├── injector/              # Swirl injector design (LCSC/GCSC models)
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
│   ├── config/                    # Configuration defaults and utilities
│   │   └── engine_config.py       # Default values and config helpers
│   └── projects/                  # Project and version management
│       ├── project.py             # Project container
│       ├── version_control.py     # Git-based design versioning
│       └── output_manager.py      # Output file organization
├── api/                           # FastAPI REST API — 42 Python files, ~3.4k lines
│   ├── main.py                    # App entry point (serves React build at /, API at /api/v1)
│   ├── routers/                   # API route handlers (13 routers)
│   │   ├── engine.py              # /engine/design, /engine/run
│   │   ├── cooling.py             # /cooling/analyze
│   │   ├── nozzle_contour.py      # /contour/generate
│   │   ├── performance.py         # /performance/analyze
│   │   ├── structural.py          # /structural/analyze
│   │   ├── feed_system.py         # /feed-system/analyze
│   │   ├── igniter.py             # /igniter/design
│   │   ├── injector.py            # /injector/design
│   │   ├── tank.py                # /tank/simulate
│   │   ├── monte_carlo.py         # /analysis/monte-carlo
│   │   ├── optimization.py        # /analysis/optimize
│   │   ├── session.py             # /session/... (state management)
│   │   └── config_io.py           # /config/import, /config/export
│   ├── models/                    # Pydantic v2 request/response models
│   │   ├── engine_models.py
│   │   ├── cooling_models.py
│   │   ├── contour_models.py
│   │   ├── performance_models.py
│   │   ├── feed_system_models.py
│   │   ├── structural_models.py
│   │   ├── session_models.py
│   │   ├── igniter_models.py
│   │   ├── injector_models.py
│   │   ├── tank_models.py
│   │   ├── monte_carlo_models.py
│   │   └── optimization_models.py
│   ├── services/                  # Utility services
│   │   ├── serialization.py       # JSON/YAML serialization
│   │   └── session_manager.py     # Session state management
│   └── tests/                     # pytest API tests (primary test location)
│       ├── conftest.py            # Shared fixtures
│       ├── test_health.py
│       ├── test_engine.py
│       ├── test_session.py
│       ├── test_monte_carlo.py
│       ├── test_igniter.py
│       ├── test_injector.py
│       ├── test_tank.py
│       └── test_optimization.py
├── web/                           # React + TypeScript frontend (Vite, served by FastAPI)
│   ├── package.json               # React 18, TypeScript 5.4, Vite 5.2, Blueprint.js
│   ├── vite.config.ts
│   └── src/
│       ├── api/                   # Axios-based API client functions
│       ├── types/                 # TypeScript type definitions (12 files matching backend)
│       ├── components/
│       │   ├── layout/            # TopBar, NavigationSidebar, StatusBar
│       │   ├── forms/             # Input forms
│       │   ├── plots/             # Plotly rendering wrapper
│       │   ├── metrics/           # KPI display components
│       │   ├── common/            # ErrorCallout, LoadingOverlay, StaleDataWarning
│       │   ├── workspace/         # Parameter panels, schematic view
│       │   └── ui/                # CommandPalette
│       ├── pages/                 # 11 design module pages
│       │   ├── EnginePage/
│       │   ├── CoolingPage/       # Includes 3D channel view + cross-section slider
│       │   ├── PerformancePage/
│       │   ├── StructuralPage/
│       │   ├── FeedSystemPage/
│       │   ├── NozzleContourPage/
│       │   ├── IgniterPage/
│       │   ├── InjectorPage/
│       │   ├── TankPage/
│       │   ├── MonteCarloPage/
│       │   └── OptimizationPage/
│       ├── store/                 # Zustand state management
│       ├── App.tsx
│       └── main.tsx
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI: API tests (Python 3.11) + web build (Node 20)
├── examples/                      # Example scripts
│   ├── new_architecture_demo.py   # v2.0 architecture showcase
│   └── 2KN_Ethanox_example.py     # Simple 2kN engine design
├── docs/                          # Documentation
├── Makefile                       # Build automation
├── pyproject.toml                 # Package configuration and tool settings
├── README.md                      # Project documentation
├── rocket_engine/                 # Legacy code (pre-v2.0, being replaced)
├── swirl_injector/                # Swirl injector standalone tool
├── torch_igniter_advanced/        # Torch igniter module + legacy tests
├── advanced_contour_design/       # 3D nozzle design tools
└── fluid_lib/                     # Fluid dynamics libraries
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
# FastAPI + React (full-stack) — single server
uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000 (serves React build)
# API docs at http://localhost:8000/docs
# All API routes under /api/v1 prefix
```

Using the Makefile:

```bash
make install-api   # pip install -e ".[dev]"
make install-web   # cd web && npm install
make dev           # run API + web dev servers in parallel
make build         # build React frontend (web/dist/)
make test-api      # run API tests (pytest api/tests/)
make lint          # ruff check . && black --check .
make format        # black . && ruff check --fix .
make clean         # remove __pycache__ and .pyc files
```

### Running Tests

```bash
# Run all tests (api/tests/ + torch_igniter_advanced/)
pytest

# API tests only (primary test suite)
pytest api/tests/ -v

# Legacy igniter tests
pytest torch_igniter_advanced/

# With coverage
pytest --cov=resa
```

### Code Quality

```bash
# Formatting (line length 100)
black --check .
black .

# Linting
ruff check .
ruff check --fix .
```

**Configured tools (pyproject.toml):**
- **Black**: line-length=100, target py39-py311
- **Ruff**: rules E, F, W, I; ignores E501 (line-length already enforced by Black)
- **pytest**: testpaths = ["api/tests", "torch_igniter_advanced"], asyncio_mode = "auto"

### CI/CD Pipeline

`.github/workflows/ci.yml` runs on every push/PR:
- **api-tests**: Python 3.11, `pytest api/tests/ -v --tb=short`, `black --check .`, `ruff check .`
- **web-build**: Node 20, `npm ci`, `tsc` (TypeScript), `vite build`

## Architecture & Design Patterns

### Key Principles

1. **Separation of Concerns** - Physics modules are pure calculations with no side effects. Solvers orchestrate physics. UI is completely decoupled from business logic.
2. **Dependency Injection** - Solvers accept optional dependencies in `__init__`, enabling mock injection for testing.
3. **Immutable Results** - All result dataclasses use `@dataclass(frozen=True)`.
4. **Interface Contracts** - Abstract base classes in `core/interfaces.py` define contracts (`Solver`, `CombustionSolver`, `CoolingSolver`, `FluidProvider`, `GeometryGenerator`, `Plotter`, `ReportGenerator`).
5. **Custom Exception Hierarchy** - All exceptions inherit from `RESAError` and carry contextual metadata (iterations, residuals, temperatures, pressures).

### Dependency Direction (strictly enforced)

```
core       → (no imports from resa/)
physics    → core
solvers    → core + physics
geometry   → core + physics
addons     → core + physics + solvers + geometry
analysis   → core + physics + solvers
visualization → core + results (never import solvers directly)
reporting  → core + results + visualization
api/       → resa package (external consumer, not part of resa/)
web/       → api/ via HTTP only
```

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
- **NumPy/SciPy** - Numerical computing
- **pandas** - Data tables for analysis results
- **numpy-stl** - STL geometry export
- **FastAPI** + **Uvicorn** - REST API server
- **Pydantic v2** - API request/response validation
- **React 18 + TypeScript 5.4** - Frontend (Blueprint.js UI, Zustand state, Axios, React Query)

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
- `EngineDashboardPlotter`, `CrossSectionPlotter`, `NozzleContourPlotter`, `GasDynamicsPlotter`, `Engine3DViewer`
- `HTMLReportGenerator`
- `MonteCarloAnalysis`, `ThrottleOptimizer`

## Guidelines for Making Changes

1. **Physics modules** (`resa/physics/`) must remain pure functions with no side effects or state. They should not import from `solvers/`.
2. **New solvers** should implement the appropriate ABC from `core/interfaces.py`.
3. **Result types** must be frozen dataclasses (`@dataclass(frozen=True)`).
4. **New addons** go in `resa/addons/<module_name>/` and should implement `AnalysisModule` for UI integration.
5. **Visualization** code goes in `resa/visualization/` using Plotly and the `PlotTheme` system.
6. **Keep the dependency direction**: `core` depends on nothing; `physics` depends on `core`; `solvers` depends on `core` + `physics`; `visualization`/`reporting` depend on everything above but never the reverse.
7. **New API endpoints**: Add a router in `api/routers/`, Pydantic models in `api/models/`, and register the router in `api/main.py`. Mirror the pattern of existing routers.
8. **REST API** (`api/`) uses FastAPI and Pydantic v2. Routers live in `api/routers/`, models in `api/models/`, services in `api/services/`. The server serves the compiled React frontend from `web/dist/` and API routes at `/api/v1`.
9. **Frontend changes**: TypeScript types in `web/src/types/` must match backend Pydantic models. UI pages live in `web/src/pages/<FeatureName>Page/`. Build with `make build` before testing the full-stack.
10. **Legacy directories** (`rocket_engine/`, `fluid_lib/`, `swirl_injector/`, `torch_igniter_advanced/`, `advanced_contour_design/`) contain older code being migrated into the `resa/` package. New development should go in `resa/`.
11. **Test files**: Primary tests are in `api/tests/` (integration tests for each router). Legacy tests are in `torch_igniter_advanced/`. New tests should use `pytest` conventions and target `resa/` package code.
12. **Format code** with Black (line-length 100) and lint with Ruff before committing. Run `make lint` to check.
