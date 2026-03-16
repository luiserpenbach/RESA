# RESA - Rocket Engine Sizing & Analysis

A comprehensive Python toolkit (v2.0.0) for liquid rocket engine preliminary design and analysis. Targets aerospace engineers working on small-to-medium liquid bipropellant engines.

**License:** MIT | **Python:** 3.9 – 3.12 | **Status:** Beta

---

## Features

- **Combustion Analysis** — CEA-based equilibrium chemistry via RocketCEA
- **Regenerative Cooling** — 1D marching solver with real fluid properties (CoolProp)
- **N2O Cooling** — Specialized two-phase N2O boiling analysis
- **Injector Design** — LCSC/GCSC swirl injector sizing with Cd estimation
- **Torch Igniter** — HEM two-phase flow igniter sizing
- **Throttle Analysis** — Operating envelope mapping and optimization
- **Performance Analysis** — Isp, thrust, and efficiency calculations
- **Structural Analysis** — Chamber and nozzle wall structural assessment
- **Feed System** — Hydraulic feed system sizing and analysis
- **Two-Phase Flow** — N2O orifice models (SPI, HEM, Dyer)
- **3D Visualization** — WebGL nozzle viewer and STL/DXF export
- **Monte Carlo UQ** — Latin Hypercube Sampling uncertainty quantification
- **Multi-point Optimization** — Scipy-based design optimization
- **Tank Simulation** — Pressurization and depletion modeling
- **REST API** — FastAPI backend with React/TypeScript frontend
- **Project Management** — Git-integrated design version control
- **HTML Reports** — Professional embedded-Plotly reports

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd RESA

# Development install (recommended)
pip install -e ".[dev]"
```

**Core dependencies:** numpy, scipy, plotly, pyyaml, CoolProp, pandas, numpy-stl, fastapi, uvicorn, pydantic

---

## Quick Start

### FastAPI + React (Full-Stack)

```bash
# Install web dependencies
cd web && npm install && npm run build && cd ..

# Start the API server (serves React frontend at http://localhost:8000)
uvicorn api.main:app --reload --port 8000
```

Using the Makefile:

```bash
make install-api      # pip install -e ".[dev]"
make install-web      # cd web && npm install
make dev              # run API + web dev servers in parallel
make build            # build React frontend
```

### Programmatic API

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
result.to_html("phoenix1_report.html")
```

### Configuration Validation

```python
config = EngineConfig(...)
validation = config.validate()

if not validation.is_valid:
    for error in validation.errors:
        print(f"ERROR: {error}")

for warning in validation.warnings:
    print(f"WARNING: {warning}")
```

### YAML Configuration

```python
config = EngineConfig.from_yaml("engine.yaml")
```

```yaml
# engine.yaml
meta:
  engine_name: "Phoenix-1"
  version: "1.0"
  designer: "Your Name"

propulsion:
  fuel: "Ethanol90"
  oxidizer: "N2O"
  thrust_n: 2200.0
  pc_bar: 25.0
  mr: 4.0
  eff_combustion: 0.95

nozzle:
  expansion_ratio: 4.1
  L_star_mm: 1200.0
  contraction_ratio: 12.0
  bell_fraction: 0.8

cooling:
  coolant: "REFPROP::NitrousOxide"
  mode: "counter-flow"
  inlet:
    pressure_bar: 97.0
    temperature_k: 298.0
  geometry:
    channel_width_throat_mm: 1.0
    channel_height_mm: 0.75
    rib_width_throat_mm: 0.6
    wall_thickness_mm: 0.5
```

---

## Architecture

```
RESA/
├── resa/                          # Main package (v2.0)
│   ├── core/                      # Config, results, interfaces, exceptions, materials, session
│   ├── physics/                   # Pure physics calculations (no side effects)
│   │   └── isentropic, heat_transfer, cooling_n2o, fluids, performance, structural, feed_system
│   ├── solvers/                   # Integrated analysis solvers
│   │   └── combustion, cooling, performance, structural, feed_system
│   ├── geometry/                  # Nozzle and channel geometry generators
│   ├── addons/                    # Specialized design modules
│   │   ├── igniter/               # Torch igniter sizing (CEA + HEM)
│   │   ├── injector/              # Swirl injector design (LCSC/GCSC)
│   │   ├── contour/               # 3D nozzle contour + STL/DXF export
│   │   └── tank/                  # Tank pressurization simulation
│   ├── analysis/                  # Monte Carlo, optimization
│   ├── visualization/             # Plotly interactive plots
│   ├── reporting/                 # HTML report generation
│   └── projects/                  # Git-based project/version management
│
├── api/                           # FastAPI REST API
│   ├── main.py                    # App entry point (serves React build)
│   ├── routers/                   # engine, cooling, nozzle_contour, performance,
│   │                              #   structural, feed_system, session, config_io
│   ├── models/                    # Pydantic v2 request/response models
│   └── services/                  # Serialization, session management
│
├── web/                           # React + TypeScript frontend (Vite, served by FastAPI)
├── examples/                      # Example scripts
├── docs/                          # Documentation
├── Makefile                       # Build automation
│
└── (legacy — being migrated into resa/)
    ├── rocket_engine/
    ├── swirl_injector/
    ├── torch_igniter_advanced/
    ├── advanced_contour_design/
    └── fluid_lib/
```

### Data Flow

```
EngineConfig (dataclass / YAML)
    → Engine.design()
        → CEASolver           (combustion)
        → NozzleGenerator     (geometry)
        → CoolingChannelGenerator (channels)
        → CoolingSolver       (thermal analysis)
    → EngineDesignResult (frozen dataclass)
        → HTMLReportGenerator (report)
        → EngineDashboardPlotter (visualization)
```

### Key Design Principles

1. **Separation of Concerns** — Physics modules are pure functions. Solvers orchestrate them. UI is fully decoupled.
2. **Dependency Injection** — Solvers accept optional deps in `__init__` for easy mock-based testing.
3. **Immutable Results** — All result dataclasses use `@dataclass(frozen=True)`.
4. **Interface Contracts** — ABCs in `core/interfaces.py` define `Solver`, `CombustionSolver`, `CoolingSolver`, `FluidProvider`, `GeometryGenerator`, etc.
5. **Custom Exceptions** — All exceptions inherit from `RESAError` with contextual metadata (iterations, residuals, temperatures, pressures).

### Dependency Direction

```
core  →  (nothing)
physics  →  core
solvers  →  core, physics
geometry  →  core, physics
addons  →  core, physics, solvers, geometry
analysis  →  core, physics, solvers
visualization / reporting  →  everything above
api  →  resa package
```

---

## Public API

```python
from resa import (
    # Core
    Engine, EngineConfig, EngineDesignResult, CombustionResult,
    # Interfaces
    Solver, FluidProvider, GeometryGenerator,
    # Exceptions
    RESAError, ConfigurationError, PhysicsError,
    # Visualization
    PlotTheme, EngineeringTheme, DarkTheme,
    EngineDashboardPlotter, CrossSectionPlotter,
    NozzleContourPlotter, GasDynamicsPlotter,
    Engine3DViewer,
    # Reporting
    HTMLReportGenerator,
    # Analysis
    MonteCarloAnalysis, ThrottleOptimizer,
)
```

### Custom Exception Handling

```python
from resa.core.exceptions import ConvergenceError, ThermodynamicError, MaterialLimitError

try:
    result = engine.design()
except ConvergenceError as e:
    print(f"Solver failed after {e.iterations} iterations (residual={e.residual:.2e})")
except ThermodynamicError as e:
    print(f"CoolProp failed: fluid={e.fluid}, P={e.pressure} Pa, T={e.temperature} K")
except MaterialLimitError as e:
    print(f"Wall too hot: {e.actual_value:.0f} K > limit {e.limit_value:.0f} K")
```

---

## Development

### Running Tests

```bash
pytest                                  # all tests
pytest torch_igniter_advanced/          # igniter-specific tests
pytest --cov=resa                       # with coverage
```

### Code Quality

```bash
black --check .     # formatting (line-length 100)
ruff check .        # linting
```

### Adding a New Addon Module

New specialized design modules go in `resa/addons/<module_name>/` and should implement the `AnalysisModule` interface from `core/interfaces.py`.

### Adding a New API Router

1. Create `api/routers/my_router.py` with FastAPI router and endpoint handlers
2. Create corresponding `api/models/my_models.py` with Pydantic v2 request/response models
3. Register the router in `api/main.py`

---

## REST API Endpoints

Base URL: `http://localhost:8000/api/v1`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/engine/design` | Run full engine design |
| POST | `/cooling/analyze` | Regenerative cooling analysis |
| POST | `/contour/generate` | Nozzle contour generation |
| POST | `/performance/analyze` | Engine performance analysis |
| POST | `/structural/analyze` | Structural analysis |
| POST | `/feed-system/analyze` | Feed system analysis |
| GET/POST | `/session/...` | Session state management |
| GET/POST | `/config/...` | Config import/export |

Full interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## References

- NASA CEA: https://cearun.grc.nasa.gov/
- CoolProp: http://www.coolprop.org/
- Humble, Henry & Larson: *Space Propulsion Analysis and Design*
- Sutton & Biblarz: *Rocket Propulsion Elements*

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Add tests for new functionality
4. Format with Black and lint with Ruff
5. Submit a pull request

## License

MIT License — see LICENSE file for details.
