"""
RESA Analysis Module
=====================

Provides uncertainty analysis and optimization tools for rocket engine design:

Monte Carlo Analysis:
- MonteCarloAnalysis: Latin Hypercube Sampling based uncertainty quantification
- Support for normal, uniform, and triangular distributions
- Parallel execution with concurrent.futures
- Sensitivity analysis (Pearson/Spearman correlations)

Monte Carlo Visualization:
- HistogramPlotter: Output distributions with P5/P50/P95 markers
- ScatterMatrixPlotter: Parameter vs output correlations
- TornadoPlotter: Sensitivity tornado charts
- MCConvergencePlotter: Running statistics vs sample count

Optimization:
- ThrottleOptimizer: Single-point optimization with constraints
- MultiPointOptimizer: Pareto optimization across throttle envelope

Optimization Visualization:
- OptConvergencePlotter: Optimization convergence history
- ParetoFrontPlotter: Multi-objective Pareto fronts
- DesignSpacePlotter: Design variable space exploration

Example:
    >>> from resa.analysis import MonteCarloAnalysis, HistogramPlotter
    >>>
    >>> mc = MonteCarloAnalysis()
    >>> mc.add_parameter('pc_bar', 25.0, 'normal', std_dev=1.0)
    >>> mc.add_parameter('mr', 4.0, 'uniform', min_val=3.5, max_val=4.5)
    >>>
    >>> result = mc.run(n_samples=1000, engine_func=my_engine, output_names=['isp'])
    >>> result.print_summary()
    >>>
    >>> hist = HistogramPlotter()
    >>> fig = hist.create_figure(result, 'isp')
    >>> fig.show()
"""

# Monte Carlo Analysis
from resa.analysis.monte_carlo import (
    MonteCarloAnalysis,
    MonteCarloResult,
    ParameterDistribution,
    run_monte_carlo,
)

# Monte Carlo Visualization
from resa.analysis.monte_carlo_plots import (
    HistogramPlotter,
    ScatterMatrixPlotter,
    TornadoPlotter,
    ConvergencePlotter as MCConvergencePlotter,
)

# Optimization
from resa.analysis.optimization import (
    ThrottleOptimizer,
    MultiPointOptimizer,
    OptimizationResult,
    DesignVariable,
    Constraint,
    OperatingPoint,
)

# Optimization Visualization
from resa.analysis.optimization_plots import (
    ConvergencePlotter as OptConvergencePlotter,
    ParetoFrontPlotter,
    DesignSpacePlotter,
)

__all__ = [
    # Monte Carlo Core
    "MonteCarloAnalysis",
    "MonteCarloResult",
    "ParameterDistribution",
    "run_monte_carlo",
    # Monte Carlo Visualization
    "HistogramPlotter",
    "ScatterMatrixPlotter",
    "TornadoPlotter",
    "MCConvergencePlotter",
    # Optimization Core
    "ThrottleOptimizer",
    "MultiPointOptimizer",
    "OptimizationResult",
    "DesignVariable",
    "Constraint",
    "OperatingPoint",
    # Optimization Visualization
    "OptConvergencePlotter",
    "ParetoFrontPlotter",
    "DesignSpacePlotter",
]
