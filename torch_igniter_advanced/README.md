# Torch Igniter Sizing Tool

A preliminary design and analysis tool for bipropellant torch igniters using Ethanol and Nitrous Oxide propellants.

## Features

- **NASA CEA Integration**: Rigorous combustion thermochemistry calculations
- **CoolProp Integration**: Accurate fluid properties for both propellants
- **L* Method Chamber Sizing**: Industry-standard approach for combustion stability
- **Simple Injector Sizing**: Orifice-based injection with proper N2O modeling
- **Performance Analysis**: Complete performance metrics including heat power output
- **JSON Configuration**: Easy configuration management and result storage

## Installation

### Prerequisites

```bash
pip install numpy scipy pandas CoolProp rocketcea
```

### Package Installation

```bash
# From repository root
pip install -e .
```

Or simply add the `torch_igniter` directory to your Python path.

## Quick Start

```python
from torch_igniter_simple import IgniterConfig, IgniterDesigner

# Create configuration
config = IgniterConfig(
    chamber_pressure=20e5,  # 20 bar
    mixture_ratio=2.0,  # O/F = 2.0
    total_mass_flow=0.050,  # 50 g/s
    ethanol_feed_pressure=25e5,
    n2o_feed_pressure=30e5,
    ethanol_feed_temperature=298.15,
    n2o_feed_temperature=293.15,
    l_star=1.0,
    expansion_ratio=3.0
)

# Design igniter
designer = IgniterDesigner()
results = designer.design(config)

# View results
print(results.summary())

# Save outputs
config.save_json("my_igniter_config.json")
results.save_json("my_igniter_results.json")
```

## Configuration Parameters

### Operating Conditions
- `chamber_pressure`: Chamber pressure (Pa)
- `mixture_ratio`: Oxidizer/Fuel mass ratio
- `total_mass_flow`: Total mass flow rate (kg/s)

### Feed System
- `ethanol_feed_pressure`: Ethanol supply pressure (Pa)
- `n2o_feed_pressure`: N2O supply pressure (Pa)
- `ethanol_feed_temperature`: Ethanol temperature (K)
- `n2o_feed_temperature`: N2O temperature (K) - critical for density

### Design Parameters
- `l_star`: Characteristic length (m), typical 0.5-1.5 for igniters
- `expansion_ratio`: Nozzle area ratio (Ae/At)
- `nozzle_type`: "conical" or "bell"
- `conical_half_angle`: Cone half angle (degrees)

### Injector
- `n2o_orifice_count`: Number of N2O injection orifices
- `ethanol_orifice_count`: Number of ethanol injection orifices
- `discharge_coefficient`: Orifice discharge coefficient, typical 0.6-0.8

## Results

The `IgniterResults` dataclass contains:

### Combustion Properties
- Flame temperature (K)
- Characteristic velocity c* (m/s)
- Specific heat ratio gamma
- Molecular weight (kg/kmol)
- **Heat power output (W)**

### Geometry
- Chamber diameter, length, volume
- Throat diameter and area
- Exit diameter and area
- Nozzle length

### Injector
- Orifice diameters for each propellant
- Injection velocities
- Pressure drops

### Performance
- Theoretical Isp (s)
- Thrust (N)
- C* efficiency

## Development Status

### Phase 1: Core Functionality ✅ COMPLETE
- [x] Configuration management
- [x] CEA integration
- [x] Basic chamber sizing
- [x] Nozzle throat calculation
- [x] Simple ethanol injector sizing
- [x] Performance metrics
- [x] Heat power calculation (LHV method)

### Phase 2: N2O HEM Implementation ✅ COMPLETE
- [x] HEM method for N2O injector sizing
- [x] Two-phase flow modeling
- [x] Choked flow detection and handling
- [x] Temperature-dependent N2O properties
- [x] Validation framework

### Phase 3: Analysis & Reporting (Next)
- [ ] Operating envelope generation
- [ ] Plotly visualizations
- [ ] HTML report generation
- [ ] Example configurations

### Phase 4: Refinement
- [ ] Extended test coverage
- [ ] Comprehensive documentation
- [ ] Parametric study tools
- [ ] Uncertainty quantification hooks

## Technical Notes

### N2O Injector Sizing (HEM Method)

**Implemented in Phase 2** with proper Homogeneous Equilibrium Model for accurate two-phase N2O flow modeling.

**Why HEM is critical:**
- N2O is stored as saturated (or near-saturated) liquid
- Undergoes phase change (flashing/cavitation) during injection
- Two-phase mixture has very different properties than pure liquid
- Simple liquid equations give 30-50% errors in orifice size

**HEM accounts for:**
- Two-phase density changes
- Vapor quality evolution
- Choked flow conditions (almost always for N2O)
- Temperature-dependent properties

**Result:** Accurate N2O orifice sizing for preliminary design

See `HEM_IMPLEMENTATION.md` for detailed technical documentation.

### Ethanol Properties

Uses CoolProp for accurate density at specified temperature and pressure. Simple incompressible flow model is adequate for liquid ethanol since it doesn't undergo phase change.

### Chamber Sizing
Uses L* method: V_chamber = L* × A_throat

Typical L* values:
- Small igniters: 0.5-0.8 m
- Medium igniters: 0.8-1.2 m
- Large igniters: 1.2-1.5 m

## Examples

See `examples/` directory for usage examples:
- `basic_example.py`: Simple baseline igniter design
- More examples coming in Phase 3

## Architecture

The tool is structured as modular components:

- `config.py`: Configuration and results dataclasses
- `cea_interface.py`: NASA CEA wrapper
- `fluids.py`: CoolProp integration
- `chamber.py`: L* method chamber sizing
- `nozzle.py`: Nozzle geometry calculations
- `performance.py`: Performance metrics
- `__init__.py`: Main design interface

## License

MIT License - See LICENSE file for details

## Contact

For questions or contributions, please contact the development team.
