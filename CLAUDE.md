# CLAUDE.md - AI Assistant Guide for RESA

## Project Overview

**RESA** (Rocket Engine Sizing & Analysis) is a Python toolkit (v2.0.0) for liquid rocket engine preliminary design and analysis. It targets aerospace engineers and covers combustion analysis, regenerative cooling, injector design, throttle analysis, two-phase flow modeling, 3D visualization, and Monte Carlo uncertainty quantification.

**License:** MIT
**Python:** 3.9 - 3.12

## Repository Structure

```
RESA/
├── resa/                          # Main package (v2.0) — 78 Python files
│   ├── __init__.py                # Public API exports
│   ├── core/                      # Configuration, results, interfaces, exceptions
│   │   ├── config.py              # EngineConfig dataclass, YAML loading, validation (~412 lines)
│   │   ├── engine.py              # Main Engine class orchestrating all solvers (542 lines)
│   │   ├── results.py             # Immutable result dataclasses (frozen=True) (290 lines)
│   │   ├── interfaces.py          # ABCs: Solver, FluidProvider, GeometryGenerator, etc. (517 lines)
│   │   └── exceptions.py          # Exception hierarchy rooted at RESAError (131 lines)
│   ├── config/                    # Alternate config module (mirrors resa/core/config.py)
│   │   ├── __init__.py
│   │   └── engine_config.py       # EngineConfig re-export / alternate entry point
│   ├── physics/                   # Pure physics calculations (no side effects)
│   │   ├── isentropic.py          # Isentropic flow relations, Mach calculations
│   │   ├── heat_transfer.py       # Bartz equation, adiabatic wall temperature
│   │   ├── cooling_n2o.py         # N2O cooling with boiling physics (~48 k lines)
│   │   └── fluids.py              # CoolProp fluid property provider
│   ├── solvers/                   # Integrated analysis solvers
│   │   ├── combustion.py          # CEASolver wrapping RocketCEA
│   │   └── cooling.py             # Regenerative cooling marching solver (~7.6 k lines)
│   ├── geometry/                  # Geometry generators
│   │   ├── nozzle.py              # Rao bell nozzle contour generation (~6.8 k lines)
│   │   └── cooling_channels.py    # Cooling channel geometry (~3.7 k lines)
│   ├── addons/                    # Specialized design modules (25 Python files)
│   │   ├── igniter/               # Torch igniter sizing (CEA, chamber, HEM)
│   │   │   ├── config.py, designer.py, chamber.py, nozzle.py
│   │   │   ├── injector.py, cea_interface.py, performance.py, fluids.py
│   │   ├── injector/              # Swirl injector design (LCSC/GCSC)
│   │   │   ├── config.py, lcsc.py, gcsc.py
│   │   │   ├── cold_flow.py, thermodynamics.py, results.py
│   │   ├── contour/               # 3D nozzle contour and STL/DXF export
│   │   │   ├── nozzle_3d.py, channels_3d.py, export.py
│   │   └── tank/                  # Tank pressurization and depletion simulation
│   │       ├── config.py, simulator.py, thermodynamics.py
│   ├── analysis/                  # Statistical and optimization tools
│   │   ├── monte_carlo.py         # Latin Hypercube Sampling uncertainty analysis (769 lines)
│   │   ├── monte_carlo_plots.py   # MC visualization (histograms, tornado, etc.)
│   │   ├── optimization.py        # Single/multi-point design optimization
│   │   └── optimization_plots.py  # Optimization convergence/Pareto plots
│   ├── visualization/             # Plotly-based interactive plots
│   │   ├── engine_plots.py        # EngineDashboardPlotter, CrossSectionPlotter, etc. (1 028 lines)
│   │   ├── engine_3d.py           # 3D WebGL nozzle viewer (Engine3DViewer)
│   │   ├── igniter_plots.py       # Torch igniter visualization
│   │   ├── injector_plots.py      # Swirl injector Cd/spray angle plots
│   │   ├── performance_plots.py   # Isp contours, throttle curves
│   │   └── themes.py              # Centralized PlotTheme system
│   ├── reporting/                 # Report generation
│   │   └── html_report.py         # Professional HTML reports with embedded Plotly (570 lines)
│   ├── ui/                        # Streamlit application
│   │   ├── app.py                 # Main Streamlit entry point (663 lines)
│   │   └── pages/                 # 12 UI page modules (see list below)
│   └── projects/                  # Project and version management
│       ├── project.py             # Project container
│       ├── version_control.py     # Git-based design versioning
│       └── output_manager.py      # Output file organization
├── api/                           # FastAPI REST API
│   ├── main.py                    # FastAPI application entry point
│   ├── models/engine_models.py    # Pydantic v2 request/response models
│   ├── routers/engine.py          # Engine design endpoints
│   ├── routers/config_io.py       # Config upload/download endpoints
│   └── services/serialization.py  # Serialization helpers
├── web/                           # React / TypeScript frontend
│   └── src/
│       ├── components/            # Plotly wrappers, form components, UI widgets
│       ├── pages/                 # Page-level components
│       ├── store/                 # State management
│       ├── api/                   # API client
│       ├── types/                 # TypeScript type definitions
│       └── App.tsx, main.tsx, router.tsx
├── examples/                      # Example scripts
│   ├── new_architecture_demo.py   # v2.0 architecture showcase
│   └── 2KN_Ethanox_example.py     # Simple 2kN engine design
├── docs/                          # Documentation
├── .streamlit/                    # Streamlit runtime configuration
├── Makefile                       # Developer convenience targets
├── pyproject.toml                 # Package configuration and tool settings
├── README.md                      # Project documentation
│
│   — Legacy directories (pre-v2.0, being migrated into resa/) —
├── rocket_engine/                 # Original monolithic design tool
├── swirl_injector/                # Swirl injector standalone tool (9 files)
├── torch_igniter_advanced/        # Torch igniter module + tests (16 files)
│                                  # Contains test_config.py, test_hem.py
├── advanced_contour_design/       # 3D nozzle design tools
└── fluid_lib/                     # Fluid dynamics libraries
```

### UI Pages (`resa/ui/pages/`)

| File | Description |
|---|---|
| `design_page.py` | Main engine design interface |
| `cooling_page.py` | Regenerative cooling analysis |
| `n2o_cooling_page.py` | N2O-specific two-phase cooling |
| `throttle_page.py` | Throttle curve analysis |
| `analysis_page.py` | Off-design analysis |
| `monte_carlo_page.py` | Monte Carlo uncertainty analysis |
| `optimization_page.py` | Design optimization |
| `igniter_page.py` | Torch igniter sizing |
| `injector_page.py` | Swirl injector design |
| `contour_page.py` | 3D nozzle contour generation |
| `tank_page.py` | Tank pressurization simulation |
| `projects_page.py` | Project and version management |

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

# FastAPI server (programmatic/REST access)
uvicorn api.main:app --reload
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

New tests should live alongside their module (e.g., `resa/core/test_config.py`) using `pytest` conventions.

### Code Quality

```bash
# Formatting (line length 100)
black --check .
black .           # apply

# Linting
ruff check .
ruff check --fix .   # auto-fix
```

**Configured tools (`pyproject.toml`):**
- **Black**: `line-length=100`, `target-version = ["py39","py310","py311"]`
- **Ruff**: rules `E, F, W, I`; ignores `E501`

## Architecture & Design Patterns

### Key Principles

1. **Separation of Concerns** — Physics modules are pure calculations with no side effects. Solvers orchestrate physics. UI is completely decoupled from business logic.
2. **Dependency Injection** — Solvers accept optional dependencies in `__init__`, enabling mock injection for testing.
3. **Immutable Results** — All result dataclasses use `@dataclass(frozen=True)`.
4. **Interface Contracts** — Abstract base classes in `core/interfaces.py` define contracts (`Solver`, `CombustionSolver`, `CoolingSolver`, `FluidProvider`, `GeometryGenerator`, `Plotter`, `ReportGenerator`, `AnalysisModule`, `MonteCarloEngine`, `Optimizer`, `VersionControl`, `OutputManager`).
5. **Custom Exception Hierarchy** — All exceptions inherit from `RESAError` and carry contextual metadata (iterations, residuals, temperatures, pressures).

### Dependency Direction (strict — do not violate)

```
core          ← depends on nothing
physics       ← depends on core
solvers       ← depends on core + physics
geometry      ← depends on core + physics
addons        ← depends on core + physics + solvers + geometry
analysis      ← depends on core + solvers
visualization ← depends on core + results
reporting     ← depends on core + results + visualization
ui            ← depends on everything above
api           ← depends on core + solvers (thin adapter layer)
```

### Data Flow

```
EngineConfig (YAML / dataclass)
    → Engine.design()
        → CEASolver (combustion via RocketCEA)
        → NozzleGenerator (Rao bell contour)
        → CoolingChannelGenerator (channel geometry)
        → CoolingSolver (thermal marching analysis)
    → EngineDesignResult (frozen dataclass)
        → HTMLReportGenerator (embedded Plotly report)
        → EngineDashboardPlotter (interactive visualization)
```

### Configuration

Engine designs are configured via `EngineConfig` dataclasses with 50+ parameters. Configs support:
- Programmatic creation via constructor
- YAML loading via `EngineConfig.from_yaml(path)`
- Validation via `config.validate()` returning `ValidationResult` with `errors` and `warnings`
- Serialization via `config.to_yaml()` and `config.to_dict()`
- Preset configs via `AnalysisPreset.demo_50n()` and `AnalysisPreset.hopper_2kn()`

Units convention in field names:
- Pressures: `_bar` suffix (e.g., `pc_bar`)
- Temperatures: `_k` suffix (Kelvin)
- Lengths: meters by default, `_mm` suffix for millimeters
- Forces: `_n` suffix (Newtons)

Key constants in `core/config.py`:
- `PROPELLANT_ALIASES` — maps common names to RocketCEA identifiers (N2O, Ethanol90, RP1, Methane, …)
- `MATERIAL_CONDUCTIVITY` — W/(m·K) for copper, inconel718, stainless, aluminum, …

### Result Dataclasses (`core/results.py`)

All frozen (`@dataclass(frozen=True)`):

| Class | Key Fields |
|---|---|
| `CombustionResult` | `pc_bar`, `mr`, `cstar`, `isp_vac`, `T_combustion`, `gamma`, `mach_exit` |
| `NozzleGeometry` | full contour arrays, radii, lengths, `theta_exit` |
| `CoolingChannelGeometry` | position arrays, dimensions, `hydraulic_diameter`, `flow_area` |
| `CoolingResult` | temperature/pressure/heat-flux arrays, `max_wall_temp`, `pressure_drop` |
| `EngineDesignResult` | all of the above + performance metrics, warnings, `to_html()` |
| `ThrottleCurve` | `List[ThrottlePoint]`, `throttle_mode`, `throttle_ratio` |
| `InjectorGeometryResult` | orifice/chamber geometry, Cd, spray angle |
| `InjectorPerformanceResult` | mass flow, pressure drop, Weber, Reynolds |

### External Dependencies

| Package | Purpose |
|---|---|
| `rocketcea` | NASA CEA equilibrium combustion calculations |
| `CoolProp` | Real fluid thermodynamic properties |
| `plotly` | Interactive visualizations |
| `streamlit` | Web UI framework |
| `fastapi` + `uvicorn` | REST API server |
| `pydantic` (v2) | API request/response validation |
| `numpy` / `scipy` | Numerical computing |
| `numpy-stl` | STL geometry export |
| `pyyaml` | YAML config loading |
| `pandas` | Tabular data handling |

## Code Conventions

### Naming

- **Classes**: `PascalCase` (`EngineConfig`, `CEASolver`, `NozzleGenerator`)
- **Functions / methods**: `snake_case` (`calculate_optimal_expansion`, `_size_throat`)
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
raise CoolingError("Station overheated", station=42, temperature=1350.0, pressure=30e5)
```

Non-fatal issues use warning classes that are **not** exceptions:
`PerformanceWarning`, `StabilityWarning`, `ThermalWarning` (all inherit from `RESAWarning`).

### Visualization Pattern

All plotters follow a consistent pattern:
- Accept data and `PlotTheme` for styling
- Provide `create_figure()` returning a Plotly `Figure`
- Provide `to_html()` for report embedding
- Use subplots for multi-panel dashboards

Plotters in `resa/visualization/engine_plots.py`:
- `EngineDashboardPlotter` — 4-panel engine dashboard
- `CrossSectionPlotter` — 2-D cross-section view
- `NozzleContourPlotter` — nozzle wall profile
- `GasDynamicsPlotter` — Mach / temperature distributions

### FastAPI / REST API Pattern

The `api/` layer is a thin adapter:
- Pydantic v2 models in `api/models/` mirror `EngineConfig` / result types
- Routers delegate directly to `Engine`, `EngineConfig`, and addon designers
- Serialization helpers in `api/services/serialization.py` convert frozen dataclasses to JSON-serialisable dicts

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

Important public exports (from `resa/__init__.py`, `__all__` has 24 items):
- `Engine`, `EngineConfig`, `EngineDesignResult`, `CombustionResult`
- `Solver`, `FluidProvider`, `GeometryGenerator` (interfaces)
- `RESAError`, `ConfigurationError`, `PhysicsError` (exceptions)
- `PlotTheme`, `EngineeringTheme`, `DarkTheme` (visualization themes)
- `EngineDashboardPlotter`, `CrossSectionPlotter`, `NozzleContourPlotter`, `GasDynamicsPlotter`
- `Engine3DViewer`
- `HTMLReportGenerator`
- `MonteCarloAnalysis`, `ThrottleOptimizer`

## Guidelines for Making Changes

1. **Physics modules** (`resa/physics/`) must remain pure functions with no side effects or state. They must not import from `solvers/`, `ui/`, or `api/`.
2. **New solvers** should implement the appropriate ABC from `core/interfaces.py`.
3. **Result types** must be frozen dataclasses (`@dataclass(frozen=True)`).
4. **New addons** go in `resa/addons/<module_name>/` and should implement `AnalysisModule` for UI integration and expose a matching Streamlit page in `resa/ui/pages/`.
5. **Visualization** code goes in `resa/visualization/` using Plotly and the `PlotTheme` system.
6. **UI pages** go in `resa/ui/pages/` as Streamlit page modules.
7. **API endpoints** go in `api/routers/` with Pydantic models in `api/models/`. Keep the layer thin — business logic lives in `resa/`, not `api/`.
8. **Keep the dependency direction** as defined in the Architecture section above.
9. **Legacy directories** (`rocket_engine/`, `fluid_lib/`, `swirl_injector/`, `torch_igniter_advanced/`, `advanced_contour_design/`) contain older code being migrated into the `resa/` package. New development should go in `resa/`.
10. **Test files** currently live in `torch_igniter_advanced/`. New tests should be co-located with their module or in a `tests/` directory using `pytest` conventions.
11. **Format code** with Black (line-length 100) and lint with Ruff before committing.
12. **`resa/config/`** exists as an alternate entry point for `EngineConfig`. Prefer importing from `resa.core.config` in new code to avoid ambiguity.
