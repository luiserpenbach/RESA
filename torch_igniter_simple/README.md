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