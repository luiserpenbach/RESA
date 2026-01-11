"""
Monte Carlo Uncertainty Analysis Engine
=======================================

Provides robust uncertainty quantification for rocket engine design using:
- Latin Hypercube Sampling (LHS) for efficient parameter space coverage
- Multiple distribution types (normal, uniform, triangular)
- Parallel execution via concurrent.futures
- Comprehensive statistics (mean, std, percentiles)
- Sensitivity analysis (Pearson and Spearman correlations)

Example:
    >>> from resa.analysis import MonteCarloAnalysis
    >>>
    >>> mc = MonteCarloAnalysis()
    >>> mc.add_parameter('pc_bar', 25.0, 'normal', std_dev=1.0)
    >>> mc.add_parameter('mr', 4.0, 'uniform', min_val=3.5, max_val=4.5)
    >>> mc.add_parameter('thrust_n', 2200, 'triangular', min_val=2100, mode=2200, max_val=2300)
    >>>
    >>> def run_engine(pc_bar, mr, thrust_n):
    ...     # Your engine analysis here
    ...     return {'isp': 250 + pc_bar * 0.5, 'thrust': thrust_n * 1.1}
    >>>
    >>> result = mc.run(n_samples=1000, engine_func=run_engine, output_names=['isp', 'thrust'])
    >>> result.print_summary()
    >>>
    >>> # Access statistics
    >>> print(f"ISP P95: {result.statistics['isp']['P95']:.1f} s")
"""

from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Any, Callable, Union, Tuple
)
from enum import Enum
import numpy as np
from scipy import stats
from scipy.stats import qmc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import warnings
import time


class DistributionType(Enum):
    """Supported probability distribution types."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"


@dataclass
class ParameterDistribution:
    """
    Defines an uncertain parameter with its probability distribution.

    Attributes:
        name: Parameter identifier (must match engine function argument)
        nominal: Nominal/baseline value
        distribution: Type of distribution ('normal', 'uniform', 'triangular')
        std_dev: Standard deviation for normal distribution
        min_val: Minimum value for uniform/triangular distributions
        max_val: Maximum value for uniform/triangular distributions
        mode: Mode (peak) for triangular distribution
    """
    name: str
    nominal: float
    distribution: str
    std_dev: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mode: Optional[float] = None

    def __post_init__(self):
        """Validate distribution parameters."""
        dist_type = self.distribution.lower()

        if dist_type == 'normal':
            if self.std_dev is None:
                raise ValueError(f"Parameter '{self.name}': Normal distribution requires std_dev")
            if self.std_dev <= 0:
                raise ValueError(f"Parameter '{self.name}': std_dev must be positive")

        elif dist_type == 'uniform':
            if self.min_val is None or self.max_val is None:
                raise ValueError(f"Parameter '{self.name}': Uniform distribution requires min_val and max_val")
            if self.min_val >= self.max_val:
                raise ValueError(f"Parameter '{self.name}': min_val must be less than max_val")

        elif dist_type == 'triangular':
            if self.min_val is None or self.max_val is None:
                raise ValueError(f"Parameter '{self.name}': Triangular distribution requires min_val and max_val")
            if self.mode is None:
                # Default mode to nominal value
                self.mode = self.nominal
            if not (self.min_val <= self.mode <= self.max_val):
                raise ValueError(f"Parameter '{self.name}': mode must be between min_val and max_val")

        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}. "
                           f"Supported: 'normal', 'uniform', 'triangular'")

    def get_scipy_distribution(self) -> stats.rv_continuous:
        """
        Get the scipy.stats distribution object.

        Returns:
            Frozen scipy distribution for sampling
        """
        dist_type = self.distribution.lower()

        if dist_type == 'normal':
            return stats.norm(loc=self.nominal, scale=self.std_dev)

        elif dist_type == 'uniform':
            loc = self.min_val
            scale = self.max_val - self.min_val
            return stats.uniform(loc=loc, scale=scale)

        elif dist_type == 'triangular':
            # scipy.stats.triang uses c = (mode - min) / (max - min)
            loc = self.min_val
            scale = self.max_val - self.min_val
            c = (self.mode - self.min_val) / scale
            return stats.triang(c=c, loc=loc, scale=scale)

    def sample_from_lhs(self, lhs_values: np.ndarray) -> np.ndarray:
        """
        Transform uniform LHS samples [0,1] to this distribution.

        Args:
            lhs_values: Uniform samples from Latin Hypercube [0, 1]

        Returns:
            Samples transformed to this distribution
        """
        dist = self.get_scipy_distribution()
        return dist.ppf(lhs_values)

    def describe(self) -> str:
        """Get human-readable description of the distribution."""
        dist_type = self.distribution.lower()

        if dist_type == 'normal':
            return f"Normal(mean={self.nominal}, std={self.std_dev})"
        elif dist_type == 'uniform':
            return f"Uniform(min={self.min_val}, max={self.max_val})"
        elif dist_type == 'triangular':
            return f"Triangular(min={self.min_val}, mode={self.mode}, max={self.max_val})"
        return f"Unknown({self.distribution})"


@dataclass
class MonteCarloResult:
    """
    Results from Monte Carlo analysis.

    Attributes:
        n_samples: Number of successful samples
        parameter_names: List of input parameter names
        output_names: List of output variable names
        input_samples: Dict of parameter_name -> sample values array
        output_samples: Dict of output_name -> result values array
        statistics: Dict of output_name -> statistics dict
        sensitivity: Dict of output_name -> sensitivity coefficients
        failed_samples: Number of failed evaluations
        elapsed_time: Total analysis time in seconds
    """
    n_samples: int
    parameter_names: List[str]
    output_names: List[str]
    input_samples: Dict[str, np.ndarray]
    output_samples: Dict[str, np.ndarray]
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    sensitivity: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    failed_samples: int = 0
    elapsed_time: float = 0.0

    def get_input_array(self) -> np.ndarray:
        """
        Get inputs as 2D array (n_samples x n_parameters).

        Returns:
            2D numpy array of input samples
        """
        return np.column_stack([self.input_samples[p] for p in self.parameter_names])

    def get_output_array(self) -> np.ndarray:
        """
        Get outputs as 2D array (n_samples x n_outputs).

        Returns:
            2D numpy array of output samples
        """
        return np.column_stack([self.output_samples[o] for o in self.output_names])

    def get_percentile(self, output_name: str, percentile: float) -> float:
        """
        Get arbitrary percentile for an output.

        Args:
            output_name: Name of the output variable
            percentile: Percentile value (0-100)

        Returns:
            Percentile value
        """
        if output_name not in self.output_samples:
            raise ValueError(f"Unknown output: {output_name}")
        return float(np.percentile(self.output_samples[output_name], percentile))

    def get_confidence_interval(
        self,
        output_name: str,
        confidence: float = 0.90
    ) -> Tuple[float, float]:
        """
        Get symmetric confidence interval.

        Args:
            output_name: Name of the output variable
            confidence: Confidence level (e.g., 0.90 for 90%)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        alpha = (1 - confidence) / 2
        lower = self.get_percentile(output_name, alpha * 100)
        upper = self.get_percentile(output_name, (1 - alpha) * 100)
        return (lower, upper)

    def print_summary(self) -> None:
        """Print formatted summary of Monte Carlo results."""
        print("=" * 70)
        print("MONTE CARLO ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Samples: {self.n_samples} successful, {self.failed_samples} failed")
        print(f"Elapsed time: {self.elapsed_time:.2f} seconds")
        print()

        # Input parameter ranges
        print("INPUT PARAMETERS:")
        print("-" * 70)
        for param in self.parameter_names:
            samples = self.input_samples[param]
            print(f"  {param:20s}: mean={np.mean(samples):10.4f}, "
                  f"std={np.std(samples):10.4f}, "
                  f"range=[{np.min(samples):.4f}, {np.max(samples):.4f}]")
        print()

        # Output statistics
        print("OUTPUT STATISTICS:")
        print("-" * 70)
        for output in self.output_names:
            stats = self.statistics.get(output, {})
            print(f"  {output}:")
            print(f"    Mean:   {stats.get('mean', np.nan):12.4f}")
            print(f"    Std:    {stats.get('std', np.nan):12.4f}")
            print(f"    P5:     {stats.get('P5', np.nan):12.4f}")
            print(f"    P50:    {stats.get('P50', np.nan):12.4f}")
            print(f"    P95:    {stats.get('P95', np.nan):12.4f}")
            print(f"    Min:    {stats.get('min', np.nan):12.4f}")
            print(f"    Max:    {stats.get('max', np.nan):12.4f}")
            print()

        # Sensitivity analysis
        if self.sensitivity:
            print("SENSITIVITY ANALYSIS (Spearman correlation):")
            print("-" * 70)
            for output in self.output_names:
                sens = self.sensitivity.get(output, {})
                spearman = sens.get('spearman', {})
                if spearman:
                    print(f"  {output}:")
                    # Sort by absolute correlation
                    sorted_params = sorted(
                        spearman.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )
                    for param, corr in sorted_params:
                        bar = "+" * int(abs(corr) * 20) if corr > 0 else "-" * int(abs(corr) * 20)
                        print(f"    {param:20s}: {corr:+.4f} {bar}")
                    print()

        print("=" * 70)

    def to_dataframe(self):
        """
        Convert results to pandas DataFrame.

        Returns:
            pandas DataFrame with all samples
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")

        data = {}
        for param in self.parameter_names:
            data[f"input_{param}"] = self.input_samples[param]
        for output in self.output_names:
            data[f"output_{output}"] = self.output_samples[output]

        return pd.DataFrame(data)


class MonteCarloAnalysis:
    """
    Monte Carlo uncertainty analysis engine.

    Uses Latin Hypercube Sampling for efficient parameter space coverage
    and supports parallel execution for performance.

    Example:
        >>> mc = MonteCarloAnalysis()
        >>> mc.add_parameter('pc_bar', 25.0, 'normal', std_dev=1.0)
        >>> mc.add_parameter('mr', 4.0, 'uniform', min_val=3.5, max_val=4.5)
        >>>
        >>> result = mc.run(
        ...     n_samples=1000,
        ...     engine_func=my_analysis,
        ...     output_names=['isp', 'thrust']
        ... )
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize Monte Carlo analysis.

        Args:
            seed: Random seed for reproducibility
        """
        self.parameters: Dict[str, ParameterDistribution] = {}
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def add_parameter(
        self,
        name: str,
        nominal: float,
        distribution: str,
        std_dev: Optional[float] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        mode: Optional[float] = None
    ) -> 'MonteCarloAnalysis':
        """
        Add an uncertain parameter to the analysis.

        Args:
            name: Parameter identifier (must match engine function argument)
            nominal: Nominal/baseline value
            distribution: 'normal', 'uniform', or 'triangular'
            std_dev: Standard deviation (for normal distribution)
            min_val: Minimum value (for uniform/triangular)
            max_val: Maximum value (for uniform/triangular)
            mode: Mode/peak value (for triangular, defaults to nominal)

        Returns:
            Self for method chaining

        Example:
            >>> mc = MonteCarloAnalysis()
            >>> mc.add_parameter('pc_bar', 25.0, 'normal', std_dev=1.0)
            >>> mc.add_parameter('mr', 4.0, 'uniform', min_val=3.5, max_val=4.5)
        """
        param = ParameterDistribution(
            name=name,
            nominal=nominal,
            distribution=distribution,
            std_dev=std_dev,
            min_val=min_val,
            max_val=max_val,
            mode=mode
        )
        self.parameters[name] = param
        return self

    def remove_parameter(self, name: str) -> 'MonteCarloAnalysis':
        """
        Remove a parameter from the analysis.

        Args:
            name: Parameter name to remove

        Returns:
            Self for method chaining
        """
        if name in self.parameters:
            del self.parameters[name]
        return self

    def clear_parameters(self) -> 'MonteCarloAnalysis':
        """
        Remove all parameters.

        Returns:
            Self for method chaining
        """
        self.parameters.clear()
        return self

    def _generate_lhs_samples(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        Generate Latin Hypercube samples for all parameters.

        Uses scipy.stats.qmc.LatinHypercube for optimal space-filling design.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Dict mapping parameter names to sample arrays
        """
        n_params = len(self.parameters)
        if n_params == 0:
            raise ValueError("No parameters defined. Add parameters with add_parameter()")

        # Generate LHS samples in unit hypercube
        sampler = qmc.LatinHypercube(d=n_params, seed=self.seed)
        lhs_uniform = sampler.random(n=n_samples)

        # Transform to parameter distributions
        samples = {}
        for i, (name, param) in enumerate(self.parameters.items()):
            samples[name] = param.sample_from_lhs(lhs_uniform[:, i])

        return samples

    def _evaluate_sample(
        self,
        sample_idx: int,
        input_dict: Dict[str, float],
        engine_func: Callable,
        output_names: List[str]
    ) -> Tuple[int, Optional[Dict[str, float]]]:
        """
        Evaluate engine function for a single sample.

        Args:
            sample_idx: Sample index for tracking
            input_dict: Parameter values for this sample
            engine_func: Function to evaluate
            output_names: Expected output names

        Returns:
            Tuple of (sample_idx, result_dict or None if failed)
        """
        try:
            result = engine_func(**input_dict)

            # Handle different return types
            if isinstance(result, dict):
                output_dict = {name: result[name] for name in output_names}
            elif hasattr(result, '__iter__') and not isinstance(result, str):
                # Assume ordered outputs
                result_list = list(result)
                if len(result_list) != len(output_names):
                    raise ValueError(f"Expected {len(output_names)} outputs, got {len(result_list)}")
                output_dict = dict(zip(output_names, result_list))
            else:
                # Single output
                if len(output_names) != 1:
                    raise ValueError(f"Expected {len(output_names)} outputs, got single value")
                output_dict = {output_names[0]: result}

            return (sample_idx, output_dict)

        except Exception as e:
            warnings.warn(f"Sample {sample_idx} failed: {str(e)}")
            return (sample_idx, None)

    def run(
        self,
        n_samples: int,
        engine_func: Callable[..., Union[Dict[str, float], List[float], float]],
        output_names: List[str],
        n_workers: Optional[int] = None,
        use_processes: bool = False,
        show_progress: bool = True
    ) -> MonteCarloResult:
        """
        Run Monte Carlo analysis.

        Args:
            n_samples: Number of samples to evaluate
            engine_func: Function that takes parameter kwargs and returns outputs.
                         Can return a dict, list/tuple (ordered), or single value.
            output_names: Names of output variables to track
            n_workers: Number of parallel workers (None = auto, 1 = sequential)
            use_processes: Use ProcessPoolExecutor instead of ThreadPoolExecutor
            show_progress: Print progress updates

        Returns:
            MonteCarloResult with all samples and statistics

        Example:
            >>> def analyze(pc_bar, mr):
            ...     return {'isp': 200 + pc_bar * 2, 'thrust': 2000 * mr / 4}
            >>>
            >>> result = mc.run(1000, analyze, ['isp', 'thrust'])
        """
        if not self.parameters:
            raise ValueError("No parameters defined. Add parameters with add_parameter()")

        if not output_names:
            raise ValueError("output_names cannot be empty")

        start_time = time.time()

        # Generate Latin Hypercube samples
        if show_progress:
            print(f"Generating {n_samples} Latin Hypercube samples for {len(self.parameters)} parameters...")

        input_samples = self._generate_lhs_samples(n_samples)
        parameter_names = list(self.parameters.keys())

        # Initialize output storage
        output_samples = {name: np.full(n_samples, np.nan) for name in output_names}
        failed_count = 0

        # Determine parallelism
        if n_workers is None:
            import os
            n_workers = min(os.cpu_count() or 4, n_samples)

        if show_progress:
            print(f"Running {n_samples} simulations with {n_workers} workers...")

        # Execute samples
        if n_workers == 1:
            # Sequential execution
            for i in range(n_samples):
                input_dict = {name: input_samples[name][i] for name in parameter_names}
                idx, result = self._evaluate_sample(i, input_dict, engine_func, output_names)

                if result is not None:
                    for name, value in result.items():
                        output_samples[name][idx] = value
                else:
                    failed_count += 1

                if show_progress and (i + 1) % max(1, n_samples // 10) == 0:
                    print(f"  Progress: {i + 1}/{n_samples} ({100 * (i + 1) / n_samples:.0f}%)")
        else:
            # Parallel execution
            Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

            with Executor(max_workers=n_workers) as executor:
                futures = []
                for i in range(n_samples):
                    input_dict = {name: input_samples[name][i] for name in parameter_names}
                    future = executor.submit(
                        self._evaluate_sample, i, input_dict, engine_func, output_names
                    )
                    futures.append(future)

                completed = 0
                for future in as_completed(futures):
                    idx, result = future.result()

                    if result is not None:
                        for name, value in result.items():
                            output_samples[name][idx] = value
                    else:
                        failed_count += 1

                    completed += 1
                    if show_progress and completed % max(1, n_samples // 10) == 0:
                        print(f"  Progress: {completed}/{n_samples} ({100 * completed / n_samples:.0f}%)")

        # Remove failed samples (NaN values)
        valid_mask = ~np.isnan(output_samples[output_names[0]])
        n_valid = np.sum(valid_mask)

        filtered_inputs = {name: samples[valid_mask] for name, samples in input_samples.items()}
        filtered_outputs = {name: samples[valid_mask] for name, samples in output_samples.items()}

        elapsed = time.time() - start_time

        if show_progress:
            print(f"Completed in {elapsed:.2f}s. {n_valid} successful, {failed_count} failed.")

        # Create result object
        result = MonteCarloResult(
            n_samples=int(n_valid),
            parameter_names=parameter_names,
            output_names=output_names,
            input_samples=filtered_inputs,
            output_samples=filtered_outputs,
            failed_samples=failed_count,
            elapsed_time=elapsed
        )

        # Compute statistics and sensitivity
        result.statistics = self.compute_statistics(result)
        result.sensitivity = self.compute_sensitivity(result)

        return result

    def compute_statistics(
        self,
        result: MonteCarloResult,
        percentiles: Optional[List[float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute statistical summary for each output.

        Args:
            result: MonteCarloResult from run()
            percentiles: Custom percentiles (default: [5, 25, 50, 75, 95])

        Returns:
            Dict mapping output names to statistics dicts
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]

        statistics = {}

        for output_name in result.output_names:
            samples = result.output_samples[output_name]

            stats_dict = {
                'mean': float(np.mean(samples)),
                'std': float(np.std(samples)),
                'var': float(np.var(samples)),
                'min': float(np.min(samples)),
                'max': float(np.max(samples)),
                'range': float(np.max(samples) - np.min(samples)),
                'skewness': float(stats.skew(samples)),
                'kurtosis': float(stats.kurtosis(samples)),
            }

            # Add percentiles
            for p in percentiles:
                stats_dict[f'P{int(p)}'] = float(np.percentile(samples, p))

            statistics[output_name] = stats_dict

        return statistics

    def compute_sensitivity(
        self,
        result: MonteCarloResult
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute sensitivity coefficients (Pearson and Spearman correlations).

        Sensitivity shows how much each input parameter affects each output.
        Higher absolute correlation = more influence.

        Args:
            result: MonteCarloResult from run()

        Returns:
            Dict of output_name -> {'pearson': {...}, 'spearman': {...}}
        """
        sensitivity = {}

        for output_name in result.output_names:
            output_samples = result.output_samples[output_name]

            pearson_coeffs = {}
            spearman_coeffs = {}

            for param_name in result.parameter_names:
                input_samples = result.input_samples[param_name]

                # Pearson correlation (linear relationship)
                pearson_r, pearson_p = stats.pearsonr(input_samples, output_samples)
                pearson_coeffs[param_name] = float(pearson_r)

                # Spearman correlation (monotonic relationship, more robust)
                spearman_r, spearman_p = stats.spearmanr(input_samples, output_samples)
                spearman_coeffs[param_name] = float(spearman_r)

            sensitivity[output_name] = {
                'pearson': pearson_coeffs,
                'spearman': spearman_coeffs
            }

        return sensitivity

    def describe_parameters(self) -> str:
        """Get summary of all defined parameters."""
        if not self.parameters:
            return "No parameters defined"

        lines = ["Defined Parameters:"]
        for name, param in self.parameters.items():
            lines.append(f"  {name}: {param.describe()}")
        return "\n".join(lines)

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get approximate bounds for each parameter.

        For normal distributions, uses +/- 3 sigma.

        Returns:
            Dict of parameter_name -> (lower, upper)
        """
        bounds = {}
        for name, param in self.parameters.items():
            dist_type = param.distribution.lower()

            if dist_type == 'normal':
                lower = param.nominal - 3 * param.std_dev
                upper = param.nominal + 3 * param.std_dev
            elif dist_type in ('uniform', 'triangular'):
                lower = param.min_val
                upper = param.max_val
            else:
                lower = param.nominal * 0.9
                upper = param.nominal * 1.1

            bounds[name] = (lower, upper)

        return bounds


# Convenience function for quick analyses
def run_monte_carlo(
    parameters: List[Dict[str, Any]],
    engine_func: Callable,
    output_names: List[str],
    n_samples: int = 1000,
    seed: Optional[int] = None,
    show_progress: bool = True
) -> MonteCarloResult:
    """
    Convenience function for quick Monte Carlo analysis.

    Args:
        parameters: List of parameter dicts with keys:
                   {'name', 'nominal', 'distribution', 'std_dev'/'min_val'/'max_val'/'mode'}
        engine_func: Function to evaluate
        output_names: Output variable names
        n_samples: Number of samples
        seed: Random seed
        show_progress: Show progress output

    Returns:
        MonteCarloResult

    Example:
        >>> result = run_monte_carlo(
        ...     parameters=[
        ...         {'name': 'pc', 'nominal': 25, 'distribution': 'normal', 'std_dev': 1},
        ...         {'name': 'mr', 'nominal': 4, 'distribution': 'uniform', 'min_val': 3.5, 'max_val': 4.5}
        ...     ],
        ...     engine_func=my_analysis,
        ...     output_names=['isp', 'thrust'],
        ...     n_samples=1000
        ... )
    """
    mc = MonteCarloAnalysis(seed=seed)

    for param in parameters:
        mc.add_parameter(**param)

    return mc.run(
        n_samples=n_samples,
        engine_func=engine_func,
        output_names=output_names,
        show_progress=show_progress
    )
