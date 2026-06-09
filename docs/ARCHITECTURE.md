# RESA Architecture

RESA (Rocket Engine Sizing & Analysis) v2.0 is a Python toolkit for liquid rocket engine preliminary design. This document describes the current architecture, design patterns, and extension points.

---

## Package Overview

```
RESA/
├── resa/          # Main Python package (~72 files, ~25k lines)
├── api/           # FastAPI REST server (~42 files, ~3.4k lines)
├── web/           # React + TypeScript frontend (Vite, Blueprint.js)
├── examples/      # Runnable example scripts
├── docs/          # Documentation
└── tests          # api/tests/ (primary) + torch_igniter_advanced/ (legacy)
```

---

## Dependency Direction

The dependency graph is strictly enforced — no layer imports from a layer above it:

```
core
  ↑
physics
  ↑
solvers ← geometry
  ↑           ↑
addons ────────
  ↑
analysis
  ↑
visualization / reporting
  ↑
api/          (external consumer of the resa package)
  ↑
web/          (HTTP only)
```

| Layer | May import from |
|-------|----------------|
| `core` | nothing (stdlib only) |
| `physics` | `core` |
| `solvers` | `core`, `physics` |
| `geometry` | `core`, `physics` |
| `addons` | `core`, `physics`, `solvers`, `geometry` |
| `analysis` | `core`, `physics`, `solvers` |
| `visualization` | `core`, result dataclasses |
| `reporting` | `core`, result dataclasses, `visualization` |
| `api/` | `resa` package (external consumer) |
| `web/` | `api/` via HTTP |

---

## Core Layer (`resa/core/`)

| Module | Purpose |
|--------|---------|
| `config.py` | `EngineConfig` dataclass (50+ fields), `ValidationResult`, `PROPELLANT_ALIASES`, `MATERIAL_CONDUCTIVITY` |
| `engine.py` | `Engine` — main orchestrator; `EngineComponents` for dependency injection |
| `results.py` | Frozen result dataclasses: `EngineDesignResult`, `CombustionResult`, `CoolingResult`, etc. |
| `interfaces.py` | Abstract base classes and protocols (see below) |
| `exceptions.py` | Custom exception hierarchy rooted at `RESAError` |
| `materials.py` | Material property definitions (thermal conductivity, yield strength) |
| `module_configs.py` | Per-module configuration helpers for the addon system |
| `session.py` | `Session` state management (used by the API session router) |

### Interfaces (`resa/core/interfaces.py`)

Every pluggable component in RESA has an ABC here. Key interfaces:

```
Solver[T]          → solve(), validate_inputs(), get_info()
CombustionSolver   → run(pc_bar, mr, eps) → CombustionResult
CoolingSolver      → solve(mdot, p_in, t_in, geometry, ...) → CoolingResult
GeometryGenerator  → generate() + optional export_dxf/export_stl
FluidProvider      → get_state(), get_state_ph(), saturation properties
Plotter            → create_figure(), to_html(), to_json(), show()
Viewer3D           → render(), to_html(), export_gltf()
ReportGenerator    → generate(), add_section()
AnalysisModule     → render_inputs(), run_analysis(), render_results() (UI plugin)
MonteCarloEngine   → add_parameter(), run(), compute_sensitivity()
Optimizer          → add_variable(), add_constraint(), set_objective(), optimize()
VersionControl     → save_version(), load_version(), diff_versions()
OutputManager      → get_output_dir(), save_result(), list_outputs()
```

### Exception Hierarchy

```
RESAError
├── ConfigurationError       invalid / incomplete engine configuration
├── ConvergenceError         solver failed (carries .iterations, .residual)
├── ThermodynamicError       CoolProp failure (carries .fluid, .pressure, .temperature)
├── GeometryError            impossible geometry
├── PhysicsError             physics calculation error
├── CombustionError          CEA failure
├── CoolingError             cooling solver failure (carries .station)
├── FlowModelError           flow model error
└── MaterialLimitError       exceeded material limit (carries .actual_value, .limit_value)

# Non-exception warnings (not raised, logged):
RESAWarning → PerformanceWarning, StabilityWarning, ThermalWarning
```

---

## Physics Layer (`resa/physics/`)

Pure functions with **no side effects and no state**. They must never import from `solvers/`.

| Module | Contents |
|--------|---------|
| `isentropic.py` | `mach_from_area_ratio()`, `get_expansion_ratio()`, `get_local_properties()` |
| `performance.py` | ISA atmosphere, `thrust_at_altitude()`, `altitude_performance_curve()`, `separation_pressure()` |
| `heat_transfer.py` | `calculate_bartz_coefficient()`, `calculate_adiabatic_wall_temp()`, `calculate_wall_temperatures()` |
| `cooling_n2o.py` | Two-phase N2O boiling heat transfer (~1,500 lines) |
| `fluids.py` | CoolProp `FluidProvider` implementation |
| `structural.py` | Thin-wall chamber/nozzle stress analysis |
| `feed_system.py` | Pipe friction, orifice ΔP, feed system hydraulics |

**Unit conventions** used in field names:
- `_bar` → pressure in bar
- `_k` → temperature in Kelvin
- `_mm` → length in millimetres
- `_n` → force in Newtons
- default → SI (Pa, m, kg, s)

---

## Solvers Layer (`resa/solvers/`)

Solvers orchestrate physics functions and maintain solver state. Each implements `Solver[T]`.

| Solver | Class | Description |
|--------|-------|-------------|
| `combustion.py` | `CEASolver` | Wraps RocketCEA; maps RESA propellant aliases to CEA strings |
| `cooling.py` | `RegenCoolingSolver` | 1D marching solver; co-flow or counter-flow modes |
| `performance.py` | `PerformanceMapSolver` | Altitude curves and throttle maps |
| `structural.py` | `StructuralSolver` | Wall stress and safety factors |
| `feed_system.py` | `FeedSystemSolver` | Line sizing, injector ΔP, pump requirements |

---

## Geometry Layer (`resa/geometry/`)

| Module | Class | Description |
|--------|-------|-------------|
| `nozzle.py` | `NozzleGenerator` | Rao bell, conical, and ideal nozzle contours |
| `cooling_channels.py` | `CoolingChannelGenerator` | Rectangular channel sizing and layout |

---

## Addons Layer (`resa/addons/`)

Standalone design modules. Each resides in its own sub-package and implements `AnalysisModule` for frontend integration.

### `igniter/` — Torch Igniter Sizing

Ethanol/N2O bipropellant torch igniter using L* chamber sizing and HEM two-phase N2O injection.

```
config.py        IgniterConfig dataclass
designer.py      IgniterDesigner.design(config) → IgniterResults
cea_interface.py CEACalculator — equilibrium combustion
chamber.py       ChamberDesigner — L* method
nozzle.py        NozzleDesigner — throat and exit sizing
injector.py      InjectorDesigner — HEM N2O orifice sizing
fluids.py        FluidProperties — CoolProp integration
performance.py   PerformanceCalculator — Isp, thrust, envelopes
```

### `injector/` — Swirl Coaxial Injector

LCSC (Liquid-Centered Swirl Coaxial) and GCSC (Gas-Centered Swirl Coaxial) sizing.

```
config.py          InjectorConfig, PropellantConfig, OperatingConditions, GeometryConfig
lcsc.py            LCSCCalculator.calculate() → InjectorResults
gcsc.py            GCSCCalculator.calculate() → InjectorResults
cold_flow.py       ColdFlowCalculator — test equivalent flow
thermodynamics.py  DischargeCoefficients, SprayAngleCorrelations, FilmThicknessCorrelations
results.py         InjectorResults, InjectorGeometry, PerformanceMetrics
```

### `contour/` — 3D Nozzle and Channel Geometry

```
nozzle_3d.py     Nozzle3DGenerator — surface-of-revolution mesh
channels_3d.py   CoolingChannel3DGenerator — helical channel mesh
export.py        export_stl_binary/ascii(), export_geometry_json(), export_mesh_obj()
```

### `tank/` — Tank Pressurization Simulation

```
config.py         TankConfig, PressurantConfig, PropellantConfig
simulator.py      TwoPhaseNitrousTank, EthanolTank — ODE integrators
thermodynamics.py Fluid property helpers
```

---

## Analysis Layer (`resa/analysis/`)

| Module | Class | Description |
|--------|-------|-------------|
| `monte_carlo.py` | `MonteCarloAnalysis` | LHS sampling, normal/uniform/triangular distributions, parallel execution, Pearson/Spearman sensitivity |
| `monte_carlo_plots.py` | `MonteCarloPlotter` | Histograms, tornado charts, correlation matrices |
| `optimization.py` | `ThrottleOptimizer` | Bounded variables, inequality/equality constraints, Nelder-Mead / DE / SLSQP |
| `optimization_plots.py` | `OptimizationPlotter` | Convergence curves, Pareto fronts |

---

## Visualization Layer (`resa/visualization/`)

All plotters follow the `Plotter` interface: `create_figure(data) → go.Figure`, `to_html() → str`, `to_json() → str`.

| Module | Class | Description |
|--------|-------|-------------|
| `themes.py` | `PlotTheme`, `EngineeringTheme`, `DarkTheme` | Centralized color/font/layout tokens |
| `engine_plots.py` | `EngineDashboardPlotter` | 4-panel dashboard: geometry+thermal, gas dynamics, coolant pressure, hydraulics |
| `engine_3d.py` | `Engine3DViewer` | WebGL nozzle viewer |
| `performance_plots.py` | `ParameterStudyPlotter`, `GasDynamicsPlotter` | Isp contours, Cf vs altitude, throttle curves |
| `igniter_plots.py` | `IgniterPlotter` | Torch igniter performance charts |
| `injector_plots.py` | `InjectorPlotter` | Cd and spray angle curves |
| `cooling_plots.py` | `CrossSectionPlotter`, `NozzleContourPlotter` | Channel cross-section slider, contour overlay |

---

## Reporting Layer (`resa/reporting/`)

```python
from resa import HTMLReportGenerator
from resa.reporting.html_report import ReportConfig

config = ReportConfig(
    include_dashboard=True,
    include_cross_section=True,
    include_3d=False,
    company_name="Acme Propulsion",
)
gen = HTMLReportGenerator(config=config)
gen.generate(engine_result, output_path="report.html")
```

Produces a self-contained HTML file with embedded Plotly charts. Print-to-PDF via browser.

---

## REST API Layer (`api/`)

FastAPI application served by Uvicorn. The single server hosts:
- React frontend at `/` (compiled from `web/dist/`)
- API at `/api/v1`
- Swagger UI at `/docs`

### Router Map

| Router | Prefix | Key Endpoints |
|--------|--------|---------------|
| `engine.py` | `/engine` | `POST /validate`, `POST /design`, `POST /parameter-study` |
| `cooling.py` | `/cooling` | `POST /analyze` |
| `nozzle_contour.py` | `/contour` | `POST /generate` |
| `performance.py` | `/performance` | `POST /analyze` |
| `structural.py` | `/structural` | `POST /analyze` |
| `feed_system.py` | `/feed-system` | `POST /analyze` |
| `igniter.py` | `/igniter` | `POST /design` |
| `injector.py` | `/injector` | `POST /design` |
| `tank.py` | `/tank` | `POST /simulate` |
| `monte_carlo.py` | `/monte-carlo` | `POST /run` (requires session_id) |
| `optimization.py` | `/optimization` | `POST /run` (requires session_id) |
| `session.py` | `/session` | `POST /create`, `GET /{id}/status`, `DELETE /{id}` |
| `config_io.py` | `/config` | `POST /import-yaml`, `POST /export-yaml` |

### Session Model

Monte Carlo and Optimization analyses operate on a session that holds an already-run `EngineDesignResult`. Workflow:

```
POST /session/create  (EngineConfigRequest)  →  { session_id, engine_result }
POST /monte-carlo/run?session_id=...         →  { statistics, sensitivity, samples }
POST /optimization/run?session_id=...        →  { optimal_variables, objective_value }
```

### Async Pattern

All compute-heavy operations run in a thread pool to avoid blocking the event loop:

```python
result = await loop.run_in_executor(None, partial(blocking_func, args))
```

---

## Frontend (`web/`)

React 18 + TypeScript 5.4 built with Vite, using Blueprint.js for UI components and Zustand for state.

| Directory | Contents |
|-----------|---------|
| `src/api/` | Axios-based API client functions (one file per router) |
| `src/types/` | TypeScript interfaces matching backend Pydantic models |
| `src/pages/` | 11 design module pages (Engine, Cooling, Performance, Structural, FeedSystem, NozzleContour, Igniter, Injector, Tank, MonteCarlo, Optimization) |
| `src/components/` | Layout (TopBar, NavigationSidebar, StatusBar), forms, plots (Plotly wrapper), metrics, workspace |
| `src/store/` | Zustand global state slices |

TypeScript types in `src/types/` must stay in sync with the corresponding Pydantic models in `api/models/`.

---

## Configuration System

`EngineConfig` is the primary configuration dataclass with 50+ fields covering:

| Group | Fields |
|-------|--------|
| Identification | `engine_name`, `version`, `designer`, `description` |
| Propellants | `fuel`, `oxidizer`, `fuel_injection_temp_k`, `oxidizer_injection_temp_k` |
| Performance | `thrust_n`, `pc_bar`, `mr`, `eff_combustion` |
| Nozzle | `expansion_ratio`, `L_star_mm`, `contraction_ratio`, `nozzle_type`, `bell_fraction` |
| Cooling | `coolant`, `coolant_mode`, `coolant_p_in_bar`, `coolant_t_in_k`, channel geometry |
| Structural | `wall_material`, `wall_thickness_mm`, safety factors |

**Supported propellant aliases:**

| RESA alias | CEA string |
|------------|-----------|
| `Ethanol90` | `Ethanol[0.866]&Water[0.134]` |
| `Ethanol80` | `Ethanol[0.800]&Water[0.200]` |
| `RP-1` | `RP-1` |
| `N2O` | `NitrousOxide` |
| `LOX` | `LOX` |

---

## Dependency Injection

`Engine` accepts an optional `EngineComponents` dataclass so any solver or provider can be swapped in testing or for custom implementations:

```python
from resa.core.engine import Engine, EngineComponents
from resa.solvers.combustion import CEASolver

components = EngineComponents(
    combustion_solver=MyCEASolver(),   # custom implementation
    fluid_provider=MockFluidProvider(),
)
engine = Engine(config, components=components)
```

---

## Testing

Primary test suite lives in `api/tests/` (pytest, asyncio mode auto):

```
conftest.py         shared fixtures (FastAPI test client, base engine config)
test_health.py      health check
test_engine.py      /engine/design, /engine/validate, /engine/parameter-study
test_session.py     /session lifecycle
test_monte_carlo.py /monte-carlo/run
test_igniter.py     /igniter/design
test_injector.py    /injector/design
test_tank.py        /tank/simulate
test_optimization.py /optimization/run
```

Run with:

```bash
pytest api/tests/ -v --tb=short   # CI command
pytest --cov=resa                 # with coverage
```

CI runs on Python 3.11 (GitHub Actions).

---

## Code Style

| Tool | Config | Rule |
|------|--------|------|
| **Black** | `line-length = 100` | Formatting |
| **Ruff** | rules E, F, W, I; ignore E501 | Linting |
| **Type hints** | Extensive throughout | `Optional`, `Dict`, `List`, `Protocol`, `Generic[T]` |
| **Docstrings** | All public modules, classes, methods | |
| **Logging** | `logging.getLogger(__name__)` | No `print()` statements |
| **Classes** | `PascalCase` | |
| **Functions** | `snake_case` | |
| **Constants** | `UPPER_SNAKE_CASE` | |
| **Private** | Leading `_` | |
