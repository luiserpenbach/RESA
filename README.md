# RESA - Rocket Engine Sizing & Analysis

A comprehensive Python toolkit for liquid rocket engine preliminary design and analysis.

## Features

- 🔥 **Combustion Analysis** - CEA-based equilibrium chemistry
- ❄️ **Regenerative Cooling** - 1D marching solver with real fluid properties
- 💉 **Injector Design** - Swirl injector sizing with Cd estimation
- 📈 **Throttle Analysis** - Operating envelope mapping
- 🔬 **Two-Phase Flow** - N2O orifice models (SPI, HEM, Dyer)
- 📊 **Interactive UI** - Streamlit-based design interface

## Installation

```bash
pip install numpy pandas scipy matplotlib CoolProp rocketcea streamlit PyYAML ezdxf plotly
```

## Quick Start

### Using the Streamlit UI

```bash
cd rocket_engine/ui
streamlit run app.py
```

### Programmatic Usage

```python
from rocket_engine.core import EngineConfig, AnalysisPreset

# Create a configuration
config = EngineConfig(
    engine_name="Hopper E2",
    fuel="Ethanol90",
    oxidizer="N2O",
    thrust_n=2200,
    pc_bar=25,
    mr=4.0,
    coolant_p_in_bar=97,
    coolant_t_in_k=298
)

# Validate configuration
validation = config.validate()
if validation.is_valid:
    print("Configuration valid!")
else:
    print(f"Errors: {validation.errors}")

# Or use a preset
preset = AnalysisPreset.hopper_2kn()
config = preset.config
```

## Architecture

```
rocket_engine/
├── core/                    # Core abstractions
│   ├── interfaces.py        # Abstract base classes / Protocols
│   ├── results.py           # All result dataclasses
│   ├── config.py            # Configuration classes
│   └── exceptions.py        # Custom exceptions
│
├── physics/                 # Pure physics calculations
│   ├── combustion.py        # CEA wrapper
│   ├── heat_transfer.py     # Bartz equation, etc.
│   ├── fluid_flow.py        # Mach relations, area-mach
│   ├── fluid_dynamics.py    # Friction factors
│   └── cooling.py           # Regen cooling solver
│
├── geometry/                # Geometry generation
│   ├── nozzle.py            # Rao bell nozzle generator
│   ├── cooling.py           # Channel geometry
│   └── injector.py          # Injector geometry
│
├── solvers/                 # Integrated solvers
│   ├── engine_solver.py     # Main engine analysis
│   └── transient_solver.py  # Startup simulation
│
├── components/              # Component models
│   ├── feed_system.py       # Lines, valves
│   └── swirl_injector/      # Injector sizing
│
├── analysis/                # Post-processing
│   ├── fluid_state.py       # Phase diagrams
│   └── performance.py       # C*, Isp maps
│
├── io/                      # Import/export
│   └── export.py            # DXF, CSV export
│
└── ui/                      # Streamlit application
    ├── app.py               # Main entry point
    └── pages/               # Individual pages
        ├── design_page.py
        ├── analysis_page.py
        ├── thermal_page.py
        ├── injector_page.py
        ├── throttle_page.py
        ├── fluids_page.py
        └── projects_page.py
```

## Key Design Principles

### 1. Separation of Concerns

- **Physics modules** contain pure calculations with no side effects
- **Solvers** orchestrate physics modules and manage state
- **Results** are immutable dataclasses
- **UI** is completely separate from business logic

### 2. Dependency Injection

```python
# Solvers accept their dependencies, enabling testing with mocks
class EngineSolver:
    def __init__(self, config, cea_solver=None, cooling_solver=None):
        self.cea = cea_solver or CEASolver(config.fuel, config.oxidizer)
        self.cooling = cooling_solver or RegenCoolingSolver(...)
```

### 3. Configuration Validation

```python
config = EngineConfig(...)
result = config.validate()

if not result.is_valid:
    for error in result.errors:
        print(f"ERROR: {error}")
    
for warning in result.warnings:
    print(f"WARNING: {warning}")
```

### 4. Custom Exceptions

```python
from rocket_engine.core.exceptions import (
    ConvergenceError,
    ThermodynamicError,
    CoolingError
)

try:
    solver.solve()
except ConvergenceError as e:
    print(f"Solver failed after {e.iterations} iterations")
except ThermodynamicError as e:
    print(f"CoolProp failed at P={e.pressure}, T={e.temperature}")
```

## Module Extension Guide

### Adding a New Analysis Module

1. Create a new page in `ui/pages/`:

```python
# ui/pages/my_analysis_page.py
def render_my_analysis_page():
    st.title("My New Analysis")
    # ... implementation
```

2. Register in `ui/pages/__init__.py`:

```python
from .my_analysis_page import render_my_analysis_page
```

3. Add navigation in `ui/app.py`:

```python
pages = {
    # ... existing pages
    '🔮 My Analysis': 'my_analysis',
}

# In the routing section:
elif page == 'my_analysis':
    from rocket_engine.ui.pages.my_analysis_page import render_my_analysis_page
    render_my_analysis_page()
```

### Adding Turbomachinery Analysis

The modular architecture supports extension. Example structure:

```
rocket_engine/
├── turbomachinery/
│   ├── __init__.py
│   ├── pump.py              # Pump performance models
│   ├── turbine.py           # Turbine analysis
│   ├── cycle_analysis.py    # Cycle thermodynamics
│   └── results.py           # TurbomachineryResult dataclass
│
└── ui/pages/
    └── turbomachinery_page.py
```

## Configuration File Format (YAML)

```yaml
meta:
  engine_name: "Hopper E2-1A"
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

## Recommendations for Future Development

### Short Term

1. **Unit Testing** - Add pytest tests for physics modules
2. **Type Hints** - Complete type annotations throughout
3. **Documentation** - Add docstrings and Sphinx docs
4. **Error Messages** - More descriptive error messages with suggestions

### Medium Term

1. **Pint Integration** - Use `pint` for unit tracking
2. **Async Support** - Async computation for UI responsiveness
3. **Caching** - Cache CEA results and CoolProp lookups
4. **Database** - SQLite for project persistence

### Long Term

1. **Optimization** - Scipy.optimize for design optimization
2. **ML Integration** - Surrogate models for rapid iteration
3. **3D CAD Export** - STEP/IGES export for CAD integration
4. **Validation Database** - Compare against test data

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## References

- NASA CEA: https://cearun.grc.nasa.gov/
- CoolProp: http://www.coolprop.org/
- Humble, Henry, & Larson: "Space Propulsion Analysis and Design"
- Sutton & Biblarz: "Rocket Propulsion Elements"
