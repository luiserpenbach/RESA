# RESA - Rocket Engine Sizing & Analysis

A comprehensive Python toolkit (v2.0.0) for liquid rocket engine preliminary design and analysis. Targets aerospace engineers working on small-to-medium liquid bipropellant engines.

**License:** MIT | **Python:** 3.9 – 3.12 | **Status:** Beta

---

## Features

- **Combustion Analysis** — CEA-based equilibrium chemistry via RocketCEA
- **Regenerative Cooling** — 1D marching solver with real fluid properties (CoolProp)
- **N2O Cooling** — Specialized two-phase N2O boiling analysis (~1.5k-line physics model)
- **Injector Design** — LCSC/GCSC swirl injector sizing with Cd and spray angle estimation
- **Torch Igniter** — HEM two-phase flow igniter sizing (Ethanol/N2O)
- **Throttle Analysis** — Operating envelope mapping via altitude performance curves
- **Performance Analysis** — Isp, thrust, Cf at arbitrary altitude (ISA model)
- **Structural Analysis** — Chamber and nozzle wall structural assessment
- **Feed System** — Hydraulic feed system sizing and pressure drop
- **Two-Phase Flow** — N2O orifice models (SPI, HEM, Dyer)
- **3D Visualization** — WebGL nozzle viewer and STL export for CAD
- **Monte Carlo UQ** — Latin Hypercube Sampling uncertainty quantification
- **Multi-point Optimization** — Scipy-based design optimization (Nelder-Mead, DE, SLSQP)
- **Tank Simulation** — Two-phase N2O and pressurized ethanol tank depletion
- **REST API** — FastAPI backend with React/TypeScript frontend
- **Project Management** — Git-integrated design version control
- **HTML Reports** — Professional embedded-Plotly standalone reports

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
# Install web dependencies and build the frontend
cd web && npm install && npm run build && cd ..

# Start the API server (serves React frontend at http://localhost:8000)
uvicorn api.main:app --reload --port 8000
```

Using the Makefile:

```bash
make install-api      # pip install -e ".[dev]"
make install-web      # cd web && npm install
make build            # build React frontend (web/dist/)
make dev              # run API + web dev servers in parallel
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

## Addon Modules

### Torch Igniter

```python
from resa.addons.igniter import IgniterDesigner, IgniterConfig

config = IgniterConfig(
    chamber_pressure=5e5,        # 5 bar [Pa]
    mixture_ratio=4.0,
    total_mass_flow=0.005,       # kg/s
    ethanol_feed_pressure=8e5,
    n2o_feed_pressure=8e5,
    ethanol_feed_temperature=293.0,
    n2o_feed_temperature=293.0,
    l_star=0.8,
    expansion_ratio=3.5,
)

result = IgniterDesigner().design(config)
print(f"Chamber diameter:  {result.chamber_diameter * 1000:.1f} mm")
print(f"Throat diameter:   {result.throat_diameter * 1000:.1f} mm")
print(f"Flame temperature: {result.flame_temperature:.0f} K")
```

### Swirl Injector (LCSC / GCSC)

```python
from resa.addons.injector import (
    InjectorConfig, PropellantConfig, OperatingConditions,
    GeometryConfig, LCSCCalculator, GCSCCalculator,
)

config = InjectorConfig(
    propellants=PropellantConfig(
        fuel="Ethanol", oxidizer="N2O",
        fuel_temperature=293.0, oxidizer_temperature=293.0,
    ),
    operating=OperatingConditions(
        inlet_pressure=30e5, pressure_drop=3e5,
        mass_flow_fuel=0.12, mass_flow_oxidizer=0.48,
        oxidizer_velocity=20.0,
    ),
    geometry=GeometryConfig(
        num_elements=7, num_fuel_ports=4, num_ox_orifices=6,
        post_thickness=0.3e-3, spray_half_angle=30.0, minimum_clearance=0.5e-3,
    ),
)

result = LCSCCalculator(config).calculate()   # or GCSCCalculator
print(f"Spray angle: {result.performance.spray_half_angle:.1f}°")
print(f"Cd:          {result.performance.discharge_coefficient:.3f}")
```

### Tank Simulation

```python
from resa.addons.tank import TwoPhaseNitrousTank, EthanolTank
from resa.addons.tank.config import TankConfig, PressurantConfig, PropellantConfig

tank_cfg = TankConfig(
    volume=0.010,                   # 10 L [m³]
    initial_liquid_mass=8.0,        # kg
    initial_ullage_pressure=50e5,   # 50 bar [Pa]
    initial_temperature=293.0,
    wall_material_properties={"density": 2700.0, "specific_heat": 900.0, "thermal_conductivity": 200.0},
    ambient_temperature=293.0,
    heat_transfer_coefficient=10.0,
)
pres_cfg = PressurantConfig(fluid_name="Nitrogen", supply_pressure=200e5, supply_temperature=293.0)
prop_cfg = PropellantConfig(fluid_name="NitrousOxide", mass_flow_rate=0.6, is_self_pressurizing=True)

sim = TwoPhaseNitrousTank(tank_cfg, pres_cfg, prop_cfg)
sol = sim.simulate(t_span=(0.0, 20.0))   # 20-second burn
```

### Monte Carlo Uncertainty Analysis

```python
from resa.analysis.monte_carlo import MonteCarloAnalysis
from resa import Engine, EngineConfig

base = EngineConfig(engine_name="Phoenix-1", fuel="Ethanol90", oxidizer="N2O",
                    thrust_n=2200, pc_bar=25, mr=4.0)

mc = MonteCarloAnalysis(seed=42)
mc.add_parameter("pc_bar", nominal=25.0, distribution="normal", std_dev=0.75)
mc.add_parameter("mr",     nominal=4.0,  distribution="normal", std_dev=0.12)

from dataclasses import replace

def engine_func(**kwargs):
    cfg = replace(base, **kwargs)
    result = Engine(cfg).design(with_cooling=False)
    return {"isp_vac": result.isp_vac, "thrust_vac": result.thrust_vac}

mc_result = mc.run(n_samples=200, engine_func=engine_func, output_names=["isp_vac", "thrust_vac"])
print(f"Isp_vac  mean={mc_result.statistics['isp_vac']['mean']:.1f} s  "
      f"P5={mc_result.statistics['isp_vac']['P5']:.1f} s  "
      f"P95={mc_result.statistics['isp_vac']['P95']:.1f} s")
```

### Design Optimization

```python
from resa.analysis.optimization import ThrottleOptimizer
from resa import Engine, EngineConfig
from dataclasses import replace

base = EngineConfig(engine_name="Phoenix-1", fuel="Ethanol90", oxidizer="N2O",
                    thrust_n=2200, pc_bar=25, mr=4.0)

optimizer = ThrottleOptimizer(method="Nelder-Mead")
optimizer.add_variable("pc_bar", min_val=15.0, max_val=40.0, initial=25.0)
optimizer.add_variable("mr",     min_val=3.0,  max_val=5.5,  initial=4.0)
optimizer.set_objective("isp_vac", minimize=False)

def eval_func(variables):
    cfg = replace(base, **variables)
    result = Engine(cfg).design(with_cooling=False)
    return {"isp_vac": result.isp_vac}

opt_result = optimizer.optimize(eval_func, max_iterations=100)
print(f"Optimal Isp: {opt_result.optimal_objective:.1f} s at {opt_result.optimal_variables}")
```

---

## Architecture

```
RESA/
├── resa/                          # Main package (v2.0) — 72 Python files, ~25k lines
│   ├── core/                      # Config, results, interfaces, exceptions, materials, session
│   ├── physics/                   # Pure physics calculations (no side effects)
│   │   └── isentropic, heat_transfer, cooling_n2o, fluids, performance, structural, feed_system
│   ├── solvers/                   # Integrated analysis solvers
│   │   └── combustion, cooling, performance, structural, feed_system
│   ├── geometry/                  # Nozzle and cooling channel geometry generators
│   ├── addons/                    # Specialized design modules
│   │   ├── igniter/               # Torch igniter sizing (CEA + HEM two-phase)
│   │   ├── injector/              # Swirl injector design (LCSC/GCSC)
│   │   ├── contour/               # 3D nozzle geometry + STL export
│   │   └── tank/                  # Two-phase N2O / pressurized ethanol tank simulation
│   ├── analysis/                  # Monte Carlo UQ, multi-point optimization
│   ├── visualization/             # Plotly interactive plots + PlotTheme system
│   ├── reporting/                 # Self-contained HTML report generation
│   └── projects/                  # Git-backed design version control
│
├── api/                           # FastAPI REST API — 42 Python files, ~3.4k lines
│   ├── main.py                    # Single-server entry point (serves React build at /)
│   ├── routers/                   # 13 routers: engine, cooling, nozzle_contour,
│   │                              #   performance, structural, feed_system, igniter,
│   │                              #   injector, tank, monte_carlo, optimization,
│   │                              #   session, config_io
│   ├── models/                    # Pydantic v2 request/response models (12 files)
│   └── services/                  # serialization, session_manager
│
├── web/                           # React 18 + TypeScript 5.4 frontend (Vite, Blueprint.js)
│   └── src/pages/                 # 11 design module pages
│
├── examples/                      # Example scripts
├── docs/                          # Documentation
├── Makefile                       # Build automation
├── pyproject.toml                 # Package config (Black, Ruff, pytest)
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
        → CEASolver               (combustion — wraps RocketCEA)
        → NozzleGenerator         (Rao bell / conical / ideal contour)
        → CoolingChannelGenerator (channel geometry)
        → CoolingSolver           (1D marching thermal analysis)
    → EngineDesignResult (frozen dataclass)
        → HTMLReportGenerator     (standalone HTML + embedded Plotly)
        → EngineDashboardPlotter  (4-panel interactive dashboard)
```

### Key Design Principles

1. **Separation of Concerns** — Physics modules are pure functions with no side effects. Solvers orchestrate them. UI is fully decoupled from physics.
2. **Dependency Injection** — `Engine` accepts optional `EngineComponents` (combustion solver, cooling solver, nozzle generator, fluid provider) enabling mock-based testing.
3. **Immutable Results** — All result dataclasses use `@dataclass(frozen=True)`.
4. **Interface Contracts** — ABCs in `core/interfaces.py` define `Solver`, `CombustionSolver`, `CoolingSolver`, `FluidProvider`, `GeometryGenerator`, `Plotter`, `ReportGenerator`, and more.
5. **Custom Exceptions** — All exceptions inherit from `RESAError` with contextual metadata (iterations, residuals, temperatures, pressures).

### Dependency Direction (strictly enforced)

```
core        →  (nothing)
physics     →  core
solvers     →  core, physics
geometry    →  core, physics
addons      →  core, physics, solvers, geometry
analysis    →  core, physics, solvers
visualization / reporting  →  core + results (never solvers directly)
api/        →  resa package (external consumer)
web/        →  api/ via HTTP only
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
pytest                          # all tests (api/tests/ + torch_igniter_advanced/)
pytest api/tests/ -v            # API tests only
pytest torch_igniter_advanced/  # legacy igniter tests
pytest --cov=resa               # with coverage
```

### Code Quality

```bash
black --check .   # formatting (line-length 100)
ruff check .      # linting
make lint         # both checks
make format       # auto-fix formatting and lints
```

### Adding a New Addon Module

1. Create `resa/addons/<module_name>/` with `config.py`, `designer.py`, and `__init__.py`
2. Implement `AnalysisModule` from `core/interfaces.py` for frontend/UI integration
3. Add a router at `api/routers/<module>.py` and Pydantic models at `api/models/<module>_models.py`
4. Register the router in `api/main.py`

### Adding a New API Router

1. Create `api/routers/my_router.py` with a FastAPI `APIRouter` and async handlers
2. Create `api/models/my_models.py` with Pydantic v2 request/response models
3. Import and include the router in `api/main.py` under `/api/v1`

---

## REST API Endpoints

Base URL: `http://localhost:8000/api/v1` | Interactive docs: `http://localhost:8000/docs`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/engine/validate` | Validate engine configuration |
| POST | `/engine/design` | Run full engine design (optional cooling) |
| POST | `/engine/parameter-study` | CEA parameter sweeps (O/F, Pc, eps) |
| POST | `/cooling/analyze` | Regenerative cooling analysis |
| POST | `/contour/generate` | Nozzle contour generation + STL export |
| POST | `/performance/analyze` | Altitude performance curves, throttle map |
| POST | `/structural/analyze` | Chamber and nozzle wall structural assessment |
| POST | `/feed-system/analyze` | Feed system hydraulic sizing |
| POST | `/igniter/design` | Torch igniter sizing (Ethanol/N2O) |
| POST | `/injector/design` | Swirl injector design (LCSC or GCSC) |
| POST | `/tank/simulate` | Propellant tank depletion simulation |
| POST | `/monte-carlo/run` | Monte Carlo uncertainty analysis |
| POST | `/optimization/run` | Design variable optimization |
| POST | `/session/create` | Create session and run initial engine design |
| GET | `/session/{id}/status` | Get module status for a session |
| DELETE | `/session/{id}` | Delete a session |
| POST | `/config/import-yaml` | Upload YAML config → JSON |
| POST | `/config/export-yaml` | Download engine config as YAML |

> Monte Carlo and Optimization endpoints require an active session (`session_id` query param).

---

## Supported Propellants

| Alias | CEA Equivalent |
|-------|---------------|
| `Ethanol90` | Ethanol 86.6% + Water 13.4% (by mass) |
| `Ethanol80` | Ethanol 80% + Water 20% |
| `RP-1` | RP-1 hydrocarbon |
| `N2O` | NitrousOxide |
| `LOX` | LOX |

---

## References

- NASA CEA: https://cearun.grc.nasa.gov/
- CoolProp: http://www.coolprop.org/
- Humble, Henry & Larson: *Space Propulsion Analysis and Design*
- Sutton & Biblarz: *Rocket Propulsion Elements*
- Nardi et al. (2014): *Dimensioning a Simplex Swirl Injector*

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Add tests for new functionality (`api/tests/` for API, `pytest` conventions)
4. Format with Black and lint with Ruff (`make lint`)
5. Submit a pull request

## License

MIT License — see LICENSE file for details.
