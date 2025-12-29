# RESA Architecture Analysis & Improvement Recommendations

## Executive Summary

RESA (Rocket Engine Sizing & Analysis) is a sophisticated Python-based engineering tool for designing, analyzing, and simulating cryogenic and storable propellant liquid rocket engines. The codebase demonstrates solid engineering principles but has room for architectural improvements that would enhance maintainability, testability, and extensibility.

---

## 1. Available Features

### 1.1 Core Engine Design & Sizing
- **Design Mode**: Sizes engine geometry to meet thrust target at design chamber pressure
- **Off-Design Mode**: Analyzes existing geometry at new Pc/MR operating points
- **Throttling Analysis**: Simulates partial oxidizer throttling with equilibrium solver
- **Throttle Curve Generation**: Creates duty cycle performance maps

### 1.2 Physics Simulation
- **Combustion Thermochemistry**: CEA equilibrium analysis via RocketCEA wrapper
- **Fluid Dynamics**: Area-Mach relations, friction factors, orifice flow models
- **Heat Transfer**: Bartz equation, adiabatic wall temperature calculation
- **Regenerative Cooling**: 1D marching algorithm with CoolProp integration
- **Transient Dynamics**: Startup simulation with ODE solver

### 1.3 Geometry Generation
- **Nozzle Contours**: Rao bell curve profiles with full chamber-nozzle assembly
- **Cooling Channels**: Constant rib/channel width strategies with helix support
- **Injector Design**: Sizing and layout tools

### 1.4 Analysis & Visualization
- **Performance Maps**: C* contours, Isp mapping, trajectory overlays
- **Phase Diagrams**: N2O T-ρ and P-T diagrams with coolant path visualization
- **2D/3D Visualization**: Cross-sections, 3D cutaway views, channel rendering
- **CAD Export**: DXF export for integration with CAD tools

### 1.5 Data Processing
- **Sensor Analysis**: Integration, FFT analysis, instability detection
- **Report Generation**: Automated specification sheets and CSV exports

### 1.6 Web Interface
- **Streamlit App**: Interactive design suite with tabs for design, off-design, and transient analysis

---

## 2. Current Architecture Assessment

### 2.1 Strengths

| Aspect | Implementation | Rating |
|--------|---------------|--------|
| **Modular Design** | Clear separation between geometry, physics, analysis, and visualization | ✅ Good |
| **Configuration-Driven** | YAML configs and `EngineConfig` dataclass centralize parameters | ✅ Good |
| **Composition Pattern** | `LiquidEngine` composes solvers rather than inheriting | ✅ Good |
| **Data Structures** | Extensive use of dataclasses for clean data flow | ✅ Good |
| **Physics Accuracy** | Well-implemented Bartz, CEA, CoolProp integration | ✅ Excellent |
| **Visualization** | Comprehensive plotting with matplotlib | ✅ Good |

### 2.2 Architectural Issues

| Issue | Severity | Location |
|-------|----------|----------|
| No dependency injection | Medium | `engine.py:168` - CEASolver hardcoded |
| Mixed concerns in `LiquidEngine` | Medium | 800+ lines with plotting, I/O, physics |
| Bare `except` clauses | Medium | Multiple files suppress errors silently |
| No type hints on many functions | Low | Throughout codebase |
| Print statements instead of logging | Low | Throughout codebase |
| No unit tests | High | Missing test directory |
| Inconsistent import paths | Medium | Relative vs absolute imports mixed |
| No interface/protocol definitions | Medium | Tight coupling between components |
| Magic numbers in physics | Low | `viscosity_gas=8.0e-5` in engine.py:332 |
| Session state coupling in Streamlit | Low | Direct mutation of engine state |

---

## 3. Detailed Improvement Recommendations

### 3.1 High Priority: Add Testing Infrastructure

**Current State**: No test directory or test files exist.

**Recommendation**: Create a comprehensive test suite.

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/
│   ├── test_combustion.py
│   ├── test_cooling.py
│   ├── test_heat_transfer.py
│   ├── test_nozzle_geometry.py
│   └── test_fluid_flow.py
├── integration/
│   ├── test_engine_design.py
│   └── test_throttle_analysis.py
└── fixtures/
    └── expected_outputs.yaml
```

**Example Test Structure**:
```python
# tests/unit/test_heat_transfer.py
import pytest
import numpy as np
from rocket_engine.src.physics.heat_transfer import calculate_bartz_coefficient

class TestBartzCoefficient:
    def test_throat_coefficient_magnitude(self):
        """Bartz coefficient at throat should be ~10,000-50,000 W/m²K for typical engines"""
        h = calculate_bartz_coefficient(
            diameters=np.array([0.024]),
            mach_numbers=np.array([1.0]),
            pc_pa=25e5,
            c_star_mps=1500,
            d_throat_m=0.024,
            T_combustion=3200,
            viscosity_gas=8e-5,
            cp_gas=2200,
            prandtl_gas=0.68,
            gamma=1.25
        )
        assert 5000 < h[0] < 100000

    def test_subsonic_vs_supersonic(self):
        """Heat transfer coefficient should be higher at throat than in chamber"""
        # Implementation here
        pass
```

### 3.2 High Priority: Introduce Dependency Injection

**Current State** (`engine.py:166-169`):
```python
class LiquidEngine:
    def __init__(self, config: EngineConfig):
        self.cfg = config
        self.cea = CEASolver(config.fuel, config.oxidizer)  # Hardcoded
```

**Recommended Approach**:
```python
from typing import Protocol

class ICombustionSolver(Protocol):
    def run(self, pc_bar: float, mr: float, eps: float) -> CombustionPoint: ...

class ICoolingSolver(Protocol):
    def solve(self, mdot: float, p_in: float, t_in: float, ...) -> dict: ...

class LiquidEngine:
    def __init__(
        self,
        config: EngineConfig,
        combustion_solver: ICombustionSolver | None = None,
        cooling_solver_factory: Callable[..., ICoolingSolver] | None = None
    ):
        self.cfg = config
        self.cea = combustion_solver or CEASolver(config.fuel, config.oxidizer)
        self._cooling_factory = cooling_solver_factory or RegenCoolingSolver
```

**Benefits**:
- Enables mocking for unit tests
- Allows swapping CEA for alternative combustion models
- Supports different cooling solver implementations

### 3.3 Medium Priority: Extract Responsibilities from `LiquidEngine`

**Current State**: `LiquidEngine` is 800+ lines handling design, analysis, throttling, plotting, and file I/O.

**Recommended Refactoring**:

```
rocket_engine/src/
├── engine/
│   ├── __init__.py
│   ├── core.py              # Core LiquidEngine (200 lines max)
│   ├── design.py            # DesignModeRunner
│   ├── analysis.py          # OffDesignAnalyzer
│   └── throttle.py          # ThrottleAnalyzer
├── io/
│   ├── specification.py     # SpecificationWriter (from save_specification)
│   └── geometry_export.py   # GeometryExporter (from export_geometry)
└── visualization/
    └── dashboard.py         # EngineDashboard (from _plot_results)
```

**Core Engine Simplified**:
```python
# engine/core.py
class LiquidEngine:
    def __init__(self, config: EngineConfig, combustion_solver: ICombustionSolver = None):
        self.cfg = config
        self.cea = combustion_solver or CEASolver(config.fuel, config.oxidizer)
        self._design_runner = DesignModeRunner(self)
        self._analyzer = OffDesignAnalyzer(self)

    def design(self, **kwargs) -> EngineDesignResult:
        return self._design_runner.run(**kwargs)

    def analyze(self, pc_bar: float, mr: float, **kwargs) -> EngineDesignResult:
        return self._analyzer.run(pc_bar, mr, **kwargs)
```

### 3.4 Medium Priority: Replace Print Statements with Logging

**Current State** (`engine.py:171`, `cooling.py:96`):
```python
print(f"--- Initializing Engine: {config.engine_name} ---")
print(f"--- Starting Cooling Solve ({mode}): {num_channels} ch...")
```

**Recommended Approach**:
```python
import logging

logger = logging.getLogger(__name__)

class LiquidEngine:
    def __init__(self, config: EngineConfig):
        logger.info("Initializing Engine: %s", config.engine_name)

# In main entry point
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

**Benefits**:
- Configurable verbosity levels
- Can route to files, external systems
- Testable (can capture log output)
- Production-ready

### 3.5 Medium Priority: Add Type Hints Throughout

**Current State** (`cooling.py:214`):
```python
def _friction_factor(self, Re, roughness, Dh):
```

**Recommended**:
```python
def _friction_factor(
    self,
    Re: float | np.ndarray,
    roughness: float,
    Dh: float
) -> float | np.ndarray:
```

**Apply to all public functions**, especially:
- `engine.py`: All public methods
- `physics/*.py`: All solver functions
- `geometry/*.py`: All generator methods

### 3.6 Medium Priority: Fix Error Handling

**Current State** (`cooling.py:201-204`):
```python
try:
    res["quality"][i] = CP.PropsSI('Q', 'H', current_H, 'P', current_P, self.fluid)
except:
    res["quality"][i] = -1
```

**Issues**:
- Bare `except` catches everything including `KeyboardInterrupt`
- No logging of the error
- Silent failure can mask bugs

**Recommended**:
```python
from CoolProp.CoolProp import PropsSI
import logging

logger = logging.getLogger(__name__)

try:
    res["quality"][i] = PropsSI('Q', 'H', current_H, 'P', current_P, self.fluid)
except ValueError as e:
    logger.debug("Quality undefined at station %d (P=%.0f, H=%.0f): %s", i, current_P, current_H, e)
    res["quality"][i] = -1  # Single-phase indicator
```

### 3.7 Low Priority: Externalize Magic Numbers

**Current State** (`engine.py:332-335`):
```python
h_g = calculate_bartz_coefficient(
    ...
    viscosity_gas=8.0e-5,  # TODO: Get from CEA if possible
    cp_gas=2200.0,
    prandtl_gas=0.68,
```

**Recommended**: Add to `EngineConfig`:
```python
@dataclass
class EngineConfig:
    # ... existing fields ...

    # Gas Properties (estimates, can be refined from CEA)
    gas_viscosity: float = 8.0e-5  # [Pa·s]
    gas_cp: float = 2200.0         # [J/kg·K]
    gas_prandtl: float = 0.68      # [-]
```

Or better, retrieve from CEA when available:
```python
class CEASolver:
    def get_transport_properties(self, pc_bar: float, mr: float) -> dict:
        """Returns viscosity, conductivity, Cp for hot gas."""
        # Query CEA transport properties
        ...
```

### 3.8 Low Priority: Standardize Import Paths

**Current State** (mixed styles):
```python
# engine.py
from src.geometry.nozzle import NozzleGenerator  # Relative-ish
from rocket_engine.src.physics.transients import TransientSimulation  # Absolute
```

**Recommended**: Use consistent absolute imports:
```python
from rocket_engine.src.geometry.nozzle import NozzleGenerator
from rocket_engine.src.physics.transients import TransientSimulation
```

Or with package restructuring:
```python
from resa.geometry import NozzleGenerator
from resa.physics import TransientSimulation
```

---

## 4. Package Structure Recommendation

### Current Structure
```
rocket_engine/
├── src/
│   ├── engine.py (800+ lines)
│   ├── physics/
│   ├── geometry/
│   └── ...
```

### Proposed Structure
```
resa/                           # Rename for clarity
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── engine.py              # Slim coordinator (~200 lines)
│   ├── config.py              # EngineConfig, EngineDesignResult
│   └── protocols.py           # Interface definitions
├── solvers/
│   ├── __init__.py
│   ├── combustion.py          # CEASolver
│   ├── cooling.py             # RegenCoolingSolver
│   └── transient.py           # TransientSimulation
├── physics/
│   ├── __init__.py
│   ├── fluid_flow.py
│   ├── heat_transfer.py
│   └── fluid_dynamics.py
├── geometry/
│   ├── __init__.py
│   ├── nozzle.py
│   ├── cooling_channels.py
│   └── injector.py
├── io/
│   ├── __init__.py
│   ├── specification.py
│   ├── dxf_export.py
│   └── yaml_config.py
├── visualization/
│   ├── __init__.py
│   ├── dashboard.py
│   ├── cross_section.py
│   ├── engine_3d.py
│   └── phase_diagrams.py
├── analysis/
│   ├── __init__.py
│   ├── performance.py
│   └── fluid_state.py
└── utils/
    ├── __init__.py
    ├── units.py
    └── logging.py
```

---

## 5. Performance Considerations

### 5.1 CoolProp Caching
**Issue**: Multiple CoolProp calls per station in cooling solver.

**Recommendation**: Batch property queries or implement caching:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_fluid_properties(fluid: str, p: float, h: float) -> tuple:
    """Cached fluid property lookup."""
    return (
        CP.PropsSI('T', 'P', p, 'H', h, fluid),
        CP.PropsSI('D', 'P', p, 'H', h, fluid),
        CP.PropsSI('V', 'P', p, 'H', h, fluid),
        CP.PropsSI('L', 'P', p, 'H', h, fluid),
        CP.PropsSI('Cpmass', 'P', p, 'H', h, fluid),
    )
```

### 5.2 NumPy Vectorization
**Issue**: Python loops in Mach number calculation (`engine.py:311-320`).

**Recommendation**: Vectorize with NumPy:
```python
# Current (slow)
machs = []
for i, ar in enumerate(area_ratios):
    m = mach_from_area_ratio(ar, gamma, supersonic=(xs[i] > 0.001))
    machs.append(m)

# Improved (vectorized where possible)
supersonic_mask = xs > 0.001
machs = np.vectorize(mach_from_area_ratio)(area_ratios, gamma, supersonic_mask)
```

---

## 6. Documentation Recommendations

### 6.1 Add Docstrings
Many functions lack docstrings. Add NumPy-style docstrings:

```python
def calculate_bartz_coefficient(
    diameters: np.ndarray,
    mach_numbers: np.ndarray,
    pc_pa: float,
    ...
) -> np.ndarray:
    """
    Calculate the Bartz heat transfer coefficient along the nozzle.

    Uses the Bartz correlation for turbulent heat transfer in rocket nozzles,
    including corrections for compressibility and wall temperature ratio.

    Parameters
    ----------
    diameters : np.ndarray
        Local nozzle diameters [m]
    mach_numbers : np.ndarray
        Local Mach numbers [-]
    pc_pa : float
        Chamber pressure [Pa]
    ...

    Returns
    -------
    np.ndarray
        Heat transfer coefficients [W/m²K]

    References
    ----------
    .. [1] Bartz, D.R., "A Simple Equation for Rapid Estimation of Rocket
           Nozzle Convective Heat Transfer Coefficients", Jet Propulsion, 1957.
    """
```

### 6.2 Add API Documentation
Generate API docs using Sphinx or MkDocs:
```
docs/
├── api/
│   ├── engine.md
│   ├── physics.md
│   └── geometry.md
├── tutorials/
│   ├── getting_started.md
│   └── design_workflow.md
└── examples/
    └── 2kn_engine.md
```

---

## 7. Priority Implementation Order

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | Add pytest infrastructure + core tests | Medium | High |
| 2 | Introduce dependency injection | Low | High |
| 3 | Add logging framework | Low | Medium |
| 4 | Extract plotting from LiquidEngine | Medium | Medium |
| 5 | Add type hints | Low | Medium |
| 6 | Fix bare except clauses | Low | Medium |
| 7 | Externalize magic numbers | Low | Low |
| 8 | Standardize imports | Low | Low |
| 9 | Add API documentation | Medium | Medium |
| 10 | Performance optimizations | Medium | Low |

---

## 8. Conclusion

RESA is a well-engineered rocket engine design tool with solid physics implementations and good modular separation. The main architectural improvements needed are:

1. **Testing**: Critical for reliability in engineering software
2. **Dependency Injection**: Enables testability and flexibility
3. **Separation of Concerns**: `LiquidEngine` should be slimmer
4. **Proper Logging**: Replace print statements
5. **Type Safety**: Add comprehensive type hints

These improvements would transform RESA from a capable research tool into a production-ready engineering platform suitable for collaborative development and long-term maintenance.

---

*Analysis generated: 2025-12-29*
*Repository: RESA (Rocket Engine Sizing & Analysis)*
