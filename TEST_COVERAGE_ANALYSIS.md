# RESA Test Coverage Analysis

## Current State

**Test coverage is approximately 2%.** The entire `resa/` package (77 Python files, 100+ classes, 200+ functions) has **zero tests**. Only 2 test files exist, both in the legacy `torch_igniter_advanced/` module:

| File | Framework | What it tests |
|------|-----------|---------------|
| `torch_igniter_advanced/test_config.py` | unittest | `IgniterConfig` validation, serialization (8 tests) |
| `torch_igniter_advanced/test_hem.py` | manual script | `InjectorDesigner` HEM orifice sizing (3 scenarios) |

## Coverage by Module

| Module | Files | Classes/Functions | Tests | Status |
|--------|-------|-------------------|-------|--------|
| `resa/core/` | 5 | 15+ classes, 30+ functions | 0 | **UNTESTED** |
| `resa/physics/` | 5 | 10+ classes, 20+ functions | 0 | **UNTESTED** |
| `resa/solvers/` | 2 | 2 classes | 0 | **UNTESTED** |
| `resa/geometry/` | 2 | 2 classes | 0 | **UNTESTED** |
| `resa/analysis/` | 3 | 15+ classes | 0 | **UNTESTED** |
| `resa/visualization/` | 6 | 15+ classes | 0 | **UNTESTED** |
| `resa/projects/` | 3 | 3 classes | 0 | **UNTESTED** |
| `resa/addons/` | 21 | 30+ classes | 0 | **UNTESTED** |
| `resa/reporting/` | 2 | 3 classes | 0 | **UNTESTED** |
| `resa/ui/` | 13 | — | 0 | **UNTESTED** |

## Recommended Test Priorities

### Priority 1 — Core correctness (highest impact)

These modules form the foundation. Bugs here silently corrupt all downstream results.

1. **`resa/physics/isentropic.py`** — Pure math functions with known analytical solutions. Easy to test, high value.
   - `get_expansion_ratio()`, `mach_from_area_ratio()`, `get_pressure_ratio()`, `get_temperature_ratio()`
   - Validate against textbook isentropic flow tables

2. **`resa/physics/heat_transfer.py`** — Bartz correlation and wall temperature calculations.
   - Test against published reference values
   - Test edge cases (throat, high Mach)

3. **`resa/core/config.py`** — `EngineConfig.validate()` guards against invalid inputs.
   - Test all validation rules (positive values, pressure ordering, propellant aliases)
   - Test YAML round-trip serialization
   - Test `AnalysisPreset` factory methods

4. **`resa/core/results.py`** — Result dataclasses used everywhere.
   - Test construction, serialization, field calculations

### Priority 2 — Solvers and geometry

5. **`resa/solvers/combustion.py`** — `CEASolver` wraps RocketCEA.
   - Test known propellant combinations against reference c*, Isp, gamma
   - Test input validation and error handling

6. **`resa/solvers/cooling.py`** — `RegenCoolingSolver` is the most complex solver.
   - Test against reference cooling data
   - Test convergence behavior and boundary conditions

7. **`resa/geometry/nozzle.py`** — Rao bell nozzle contour generation.
   - Test throat/exit dimensions match inputs
   - Test contour smoothness and monotonicity

8. **`resa/geometry/cooling_channels.py`** — Channel layout generation.
   - Test channel count and dimensions at key stations

### Priority 3 — N2O two-phase physics

9. **`resa/physics/cooling_n2o.py`** — Large module (10+ classes) with complex two-phase flow physics.
   - Test N2O property functions against NIST data
   - Test saturation properties, flow regime detection
   - Test boiling/condensing heat transfer correlations
   - Test two-phase pressure drop models

10. **`resa/physics/fluids.py`** — CoolProp wrapper.
    - Test fluid states against known thermodynamic data

### Priority 4 — Analysis tools

11. **`resa/analysis/monte_carlo.py`** — Uncertainty quantification.
    - Test distribution sampling, statistics, convergence

12. **`resa/analysis/optimization.py`** — Design optimization.
    - Test constraint evaluation, objective functions, Pareto front generation

### Priority 5 — Addons

13. **`resa/addons/injector/`** — Swirl injector sizing (GCSC/LCSC methods).
    - Test thermodynamic calculations, discharge coefficients, spray angle correlations

14. **`resa/addons/tank/`** — Tank blowdown simulation.
    - Test N2O saturation properties, tank draining transients

15. **`resa/addons/contour/`** — 3D geometry and STL export.
    - Test mesh generation, STL export round-trip

### Lower priority

- **`resa/core/engine.py`** — Integration test of the full design pipeline. Depends on solvers being testable first. Use mocks for CEA/CoolProp.
- **`resa/visualization/`** — Plotting code. Test that functions run without error; visual correctness is hard to automate.
- **`resa/ui/`** — Streamlit pages. Better tested via end-to-end/manual testing.
- **`resa/projects/`** — Project management, version control. Test file I/O and serialization.

## Quick Wins

These require minimal setup and provide immediate value:

1. **Isentropic relations** — Pure functions, textbook answers available
2. **Config validation** — Dataclass validation, no external dependencies
3. **Result dataclasses** — Construction and serialization tests
4. **Exception hierarchy** — Verify inheritance and string representations
5. **Nozzle geometry** — Verify dimensions and contour properties

## Recommended Test Structure

```
tests/
├── core/
│   ├── test_config.py
│   ├── test_engine.py
│   ├── test_exceptions.py
│   └── test_results.py
├── physics/
│   ├── test_isentropic.py
│   ├── test_heat_transfer.py
│   ├── test_fluids.py
│   └── test_cooling_n2o.py
├── solvers/
│   ├── test_combustion.py
│   └── test_cooling.py
├── geometry/
│   ├── test_nozzle.py
│   └── test_cooling_channels.py
├── analysis/
│   ├── test_monte_carlo.py
│   └── test_optimization.py
└── addons/
    ├── test_injector.py
    ├── test_tank.py
    └── test_contour.py
```
