"""
Multi-Point Optimization Framework for Rocket Engine Design.

Provides optimization tools for:
- Single-point throttle optimization with constraints
- Multi-point Pareto optimization across throttle envelope
- Design space exploration and sensitivity analysis

Uses scipy.optimize for numerical optimization with support for:
- Bounded design variables
- Inequality and equality constraints
- Multiple optimization algorithms (Nelder-Mead, differential evolution, SLSQP)

Example:
    >>> optimizer = ThrottleOptimizer()
    >>> optimizer.add_variable('pc', min_val=10, max_val=50, initial=25)
    >>> optimizer.add_variable('mr', min_val=1.5, max_val=3.0, initial=2.2)
    >>> optimizer.add_constraint('T_wall_max', 'pc', 'max', 800)
    >>> optimizer.set_objective('thrust', minimize=False)
    >>>
    >>> def eval_func(variables):
    ...     pc, mr = variables['pc'], variables['mr']
    ...     return {'thrust': compute_thrust(pc, mr), 'T_wall_max': compute_temp(pc, mr)}
    >>>
    >>> result = optimizer.optimize(eval_func, max_iterations=100)
    >>> print(f"Optimal: pc={result.optimal_variables['pc']:.1f} bar")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from enum import Enum
import numpy as np
from datetime import datetime
import warnings

try:
    from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
    from scipy.optimize import OptimizeResult as ScipyOptimizeResult
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Optimization features will be limited.")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ConstraintType(Enum):
    """Type of constraint."""
    MIN = "min"      # value >= limit
    MAX = "max"      # value <= limit
    EQ = "eq"        # value == limit (with tolerance)


class ObjectiveDirection(Enum):
    """Optimization direction."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class DesignVariable:
    """
    Design variable for optimization.

    Attributes:
        name: Variable identifier (e.g., 'pc', 'mr', 'channel_height')
        min_val: Lower bound
        max_val: Upper bound
        initial: Starting value for optimization
        scale: Scaling factor for normalization (optional)
    """
    name: str
    min_val: float
    max_val: float
    initial: float
    scale: Optional[float] = None

    def __post_init__(self):
        """Validate variable bounds."""
        if self.min_val >= self.max_val:
            raise ValueError(f"min_val ({self.min_val}) must be < max_val ({self.max_val})")
        if not (self.min_val <= self.initial <= self.max_val):
            raise ValueError(f"initial ({self.initial}) must be within bounds [{self.min_val}, {self.max_val}]")
        if self.scale is None:
            self.scale = self.max_val - self.min_val

    @property
    def bounds(self) -> Tuple[float, float]:
        """Get bounds tuple for scipy."""
        return (self.min_val, self.max_val)

    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1] range."""
        return (value - self.min_val) / (self.max_val - self.min_val)

    def denormalize(self, normalized: float) -> float:
        """Denormalize from [0, 1] to actual range."""
        return normalized * (self.max_val - self.min_val) + self.min_val


@dataclass
class Constraint:
    """
    Optimization constraint.

    Attributes:
        name: Constraint identifier
        variable: Output variable this constraint applies to
        type: Constraint type (min, max, or eq)
        value: Constraint limit value
        tolerance: Tolerance for equality constraints
    """
    name: str
    variable: str
    type: ConstraintType
    value: float
    tolerance: float = 1e-6

    def evaluate(self, outputs: Dict[str, float]) -> float:
        """
        Evaluate constraint violation.

        Returns:
            Violation amount (negative means feasible for inequality constraints)
        """
        if self.variable not in outputs:
            raise KeyError(f"Constraint variable '{self.variable}' not in outputs")

        actual = outputs[self.variable]

        if self.type == ConstraintType.MIN:
            # value >= limit -> violation = limit - value
            return self.value - actual
        elif self.type == ConstraintType.MAX:
            # value <= limit -> violation = value - limit
            return actual - self.value
        else:  # EQ
            return abs(actual - self.value) - self.tolerance

    def is_satisfied(self, outputs: Dict[str, float]) -> bool:
        """Check if constraint is satisfied."""
        return self.evaluate(outputs) <= 0


@dataclass
class OperatingPoint:
    """
    Operating point for multi-point optimization.

    Attributes:
        name: Point identifier (e.g., 'full_thrust', 'deep_throttle')
        weight: Importance weight for Pareto optimization
        pc_range: Chamber pressure range [min, max] in bar
        mr_range: Mixture ratio range [min, max]
        constraints: Additional constraints for this point
    """
    name: str
    weight: float
    pc_range: Tuple[float, float]
    mr_range: Tuple[float, float]
    constraints: List[Constraint] = field(default_factory=list)

    def __post_init__(self):
        """Validate operating point."""
        if self.weight <= 0:
            raise ValueError(f"Weight must be positive, got {self.weight}")
        if self.pc_range[0] >= self.pc_range[1]:
            raise ValueError(f"Invalid pc_range: {self.pc_range}")
        if self.mr_range[0] >= self.mr_range[1]:
            raise ValueError(f"Invalid mr_range: {self.mr_range}")


@dataclass
class OptimizationHistory:
    """
    Tracks optimization progress.

    Stores iteration-by-iteration data for convergence analysis.
    """
    iterations: List[int] = field(default_factory=list)
    objective_values: List[float] = field(default_factory=list)
    variable_values: List[Dict[str, float]] = field(default_factory=list)
    constraint_violations: List[Dict[str, float]] = field(default_factory=list)
    feasible: List[bool] = field(default_factory=list)

    def add_iteration(
        self,
        iteration: int,
        objective: float,
        variables: Dict[str, float],
        violations: Dict[str, float],
        is_feasible: bool
    ):
        """Record an iteration."""
        self.iterations.append(iteration)
        self.objective_values.append(objective)
        self.variable_values.append(variables.copy())
        self.constraint_violations.append(violations.copy())
        self.feasible.append(is_feasible)

    @property
    def best_feasible_iteration(self) -> Optional[int]:
        """Get iteration with best feasible objective."""
        feasible_indices = [i for i, f in enumerate(self.feasible) if f]
        if not feasible_indices:
            return None
        objectives = [self.objective_values[i] for i in feasible_indices]
        best_idx = feasible_indices[np.argmin(objectives)]
        return self.iterations[best_idx]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'iterations': self.iterations,
            'objective_values': self.objective_values,
            'variable_values': self.variable_values,
            'constraint_violations': self.constraint_violations,
            'feasible': self.feasible,
        }


@dataclass
class OptimizationResult:
    """
    Result from optimization run.

    Attributes:
        success: Whether optimization converged successfully
        optimal_variables: Best design variable values
        optimal_objective: Best objective function value
        optimal_outputs: All outputs at optimal point
        history: Full optimization history
        message: Status message from optimizer
        n_iterations: Number of iterations performed
        n_evaluations: Number of function evaluations
        constraints_satisfied: Whether all constraints are met
    """
    success: bool
    optimal_variables: Dict[str, float]
    optimal_objective: float
    optimal_outputs: Dict[str, float]
    history: OptimizationHistory
    message: str
    n_iterations: int
    n_evaluations: int
    constraints_satisfied: bool
    timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 50,
            "OPTIMIZATION RESULT",
            "=" * 50,
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
            f"Message: {self.message}",
            f"Iterations: {self.n_iterations}",
            f"Function evaluations: {self.n_evaluations}",
            f"Constraints satisfied: {self.constraints_satisfied}",
            "",
            "Optimal Variables:",
        ]
        for name, value in self.optimal_variables.items():
            lines.append(f"  {name}: {value:.6g}")

        lines.extend([
            "",
            f"Optimal Objective: {self.optimal_objective:.6g}",
            "",
            "Optimal Outputs:",
        ])
        for name, value in self.optimal_outputs.items():
            lines.append(f"  {name}: {value:.6g}")

        lines.append("=" * 50)
        return "\n".join(lines)


@dataclass
class ParetoPoint:
    """Single point on a Pareto front."""
    objectives: Dict[str, float]
    variables: Dict[str, float]
    outputs: Dict[str, float]
    operating_point: Optional[str] = None

    def dominates(self, other: 'ParetoPoint', directions: Dict[str, bool]) -> bool:
        """
        Check if this point dominates another.

        Args:
            other: Another Pareto point
            directions: Dict mapping objective names to True if minimizing

        Returns:
            True if this point dominates other
        """
        dominated = False
        for obj_name, minimize in directions.items():
            self_val = self.objectives[obj_name]
            other_val = other.objectives[obj_name]

            if minimize:
                if self_val > other_val:
                    return False
                if self_val < other_val:
                    dominated = True
            else:
                if self_val < other_val:
                    return False
                if self_val > other_val:
                    dominated = True

        return dominated


@dataclass
class ParetoResult:
    """
    Result from multi-objective Pareto optimization.

    Attributes:
        pareto_front: List of non-dominated points
        all_points: All evaluated points
        operating_points: Operating points used
        objectives: Objective names and directions
        history: Optimization history per objective
    """
    pareto_front: List[ParetoPoint]
    all_points: List[ParetoPoint]
    operating_points: List[OperatingPoint]
    objectives: Dict[str, bool]  # name -> minimize
    history: Dict[str, OptimizationHistory]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def n_pareto_points(self) -> int:
        """Number of points on Pareto front."""
        return len(self.pareto_front)

    def get_objective_arrays(self) -> Dict[str, np.ndarray]:
        """Get objective values as numpy arrays."""
        result = {}
        for obj_name in self.objectives.keys():
            result[obj_name] = np.array([p.objectives[obj_name] for p in self.pareto_front])
        return result

    def get_best_for_objective(self, objective: str) -> ParetoPoint:
        """Get point that is best for a specific objective."""
        minimize = self.objectives.get(objective, True)
        if minimize:
            return min(self.pareto_front, key=lambda p: p.objectives[objective])
        else:
            return max(self.pareto_front, key=lambda p: p.objectives[objective])


# =============================================================================
# THROTTLE OPTIMIZER (Single-Point)
# =============================================================================

class ThrottleOptimizer:
    """
    Single-point constrained optimization for rocket engine design.

    Optimizes design variables to minimize/maximize an objective
    while satisfying constraints on other outputs.

    Example:
        >>> optimizer = ThrottleOptimizer()
        >>> optimizer.add_variable('pc', 10, 50, 25)
        >>> optimizer.add_variable('mr', 1.5, 3.0, 2.2)
        >>> optimizer.add_constraint('max_temp', 'T_wall_max', 'max', 800)
        >>> optimizer.set_objective('thrust', minimize=False)
        >>>
        >>> result = optimizer.optimize(eval_engine, max_iterations=100)
    """

    def __init__(self, method: str = 'SLSQP'):
        """
        Initialize optimizer.

        Args:
            method: Scipy optimization method. Options:
                - 'SLSQP': Sequential Least Squares Programming (default, handles constraints)
                - 'L-BFGS-B': Limited-memory BFGS with bounds
                - 'differential_evolution': Global optimization
                - 'Nelder-Mead': Simplex method (no gradient needed)
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for optimization. Install with: pip install scipy")

        self.method = method
        self.variables: Dict[str, DesignVariable] = {}
        self.constraints: Dict[str, Constraint] = {}
        self.objective_name: Optional[str] = None
        self.objective_minimize: bool = True
        self.history: OptimizationHistory = OptimizationHistory()
        self._iteration_count: int = 0
        self._eval_count: int = 0

    def add_variable(
        self,
        name: str,
        min_val: float,
        max_val: float,
        initial: float,
        scale: Optional[float] = None
    ) -> 'ThrottleOptimizer':
        """
        Add a design variable.

        Args:
            name: Variable identifier
            min_val: Lower bound
            max_val: Upper bound
            initial: Starting value
            scale: Optional scaling factor

        Returns:
            Self for method chaining
        """
        self.variables[name] = DesignVariable(
            name=name,
            min_val=min_val,
            max_val=max_val,
            initial=initial,
            scale=scale
        )
        return self

    def add_constraint(
        self,
        name: str,
        variable: str,
        constraint_type: Union[str, ConstraintType],
        value: float,
        tolerance: float = 1e-6
    ) -> 'ThrottleOptimizer':
        """
        Add an optimization constraint.

        Args:
            name: Constraint identifier
            variable: Output variable to constrain
            constraint_type: 'min', 'max', or 'eq'
            value: Constraint limit value
            tolerance: Tolerance for equality constraints

        Returns:
            Self for method chaining
        """
        if isinstance(constraint_type, str):
            constraint_type = ConstraintType(constraint_type)

        self.constraints[name] = Constraint(
            name=name,
            variable=variable,
            type=constraint_type,
            value=value,
            tolerance=tolerance
        )
        return self

    def set_objective(
        self,
        name: str,
        minimize: bool = True
    ) -> 'ThrottleOptimizer':
        """
        Set the objective function.

        Args:
            name: Output variable to optimize
            minimize: True to minimize, False to maximize

        Returns:
            Self for method chaining
        """
        self.objective_name = name
        self.objective_minimize = minimize
        return self

    def _array_to_dict(self, x: np.ndarray) -> Dict[str, float]:
        """Convert numpy array to variable dictionary."""
        var_names = list(self.variables.keys())
        return {name: x[i] for i, name in enumerate(var_names)}

    def _dict_to_array(self, d: Dict[str, float]) -> np.ndarray:
        """Convert variable dictionary to numpy array."""
        var_names = list(self.variables.keys())
        return np.array([d[name] for name in var_names])

    def _get_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds list for scipy."""
        return [var.bounds for var in self.variables.values()]

    def _get_initial(self) -> np.ndarray:
        """Get initial values array."""
        return np.array([var.initial for var in self.variables.values()])

    def optimize(
        self,
        eval_func: Callable[[Dict[str, float]], Dict[str, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = False
    ) -> OptimizationResult:
        """
        Run optimization.

        Args:
            eval_func: Function that takes variable dict and returns output dict.
                       Must return all constrained and objective variables.
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress

        Returns:
            OptimizationResult with optimal values and history
        """
        if self.objective_name is None:
            raise ValueError("Objective not set. Call set_objective() first.")

        if len(self.variables) == 0:
            raise ValueError("No design variables defined. Call add_variable() first.")

        # Reset state
        self.history = OptimizationHistory()
        self._iteration_count = 0
        self._eval_count = 0
        best_outputs: Dict[str, float] = {}

        def objective_wrapper(x: np.ndarray) -> float:
            """Wrapper for scipy optimizer."""
            self._eval_count += 1

            # Convert to dict
            var_dict = self._array_to_dict(x)

            # Evaluate
            outputs = eval_func(var_dict)

            # Store for constraint checking
            nonlocal best_outputs
            best_outputs = outputs

            # Get objective value
            if self.objective_name not in outputs:
                raise KeyError(f"Objective '{self.objective_name}' not in eval_func outputs")

            obj_value = outputs[self.objective_name]

            # Check constraints
            violations = {}
            feasible = True
            for name, constraint in self.constraints.items():
                violation = constraint.evaluate(outputs)
                violations[name] = violation
                if violation > 0:
                    feasible = False

            # Record history
            self._iteration_count += 1
            display_obj = obj_value if self.objective_minimize else -obj_value
            self.history.add_iteration(
                self._iteration_count,
                display_obj,
                var_dict,
                violations,
                feasible
            )

            if verbose and self._iteration_count % 10 == 0:
                status = "FEASIBLE" if feasible else "INFEASIBLE"
                print(f"Iter {self._iteration_count}: obj={display_obj:.4g} [{status}]")

            # Negate for maximization
            return obj_value if self.objective_minimize else -obj_value

        # Build constraint functions for scipy
        scipy_constraints = []
        for name, constraint in self.constraints.items():
            if constraint.type == ConstraintType.MIN:
                # g(x) >= 0 form for scipy
                def constraint_func(x, c=constraint):
                    outputs = eval_func(self._array_to_dict(x))
                    return outputs[c.variable] - c.value
                scipy_constraints.append({
                    'type': 'ineq',
                    'fun': constraint_func
                })
            elif constraint.type == ConstraintType.MAX:
                def constraint_func(x, c=constraint):
                    outputs = eval_func(self._array_to_dict(x))
                    return c.value - outputs[c.variable]
                scipy_constraints.append({
                    'type': 'ineq',
                    'fun': constraint_func
                })
            else:  # EQ
                def constraint_func(x, c=constraint):
                    outputs = eval_func(self._array_to_dict(x))
                    return outputs[c.variable] - c.value
                scipy_constraints.append({
                    'type': 'eq',
                    'fun': constraint_func
                })

        # Run optimization
        bounds = self._get_bounds()
        x0 = self._get_initial()

        if self.method == 'differential_evolution':
            result = differential_evolution(
                objective_wrapper,
                bounds=bounds,
                maxiter=max_iterations,
                tol=tolerance,
                seed=42,
                polish=True
            )
        else:
            options = {
                'maxiter': max_iterations,
                'ftol': tolerance,
                'disp': verbose
            }

            result = minimize(
                objective_wrapper,
                x0,
                method=self.method,
                bounds=bounds,
                constraints=scipy_constraints if scipy_constraints else None,
                options=options
            )

        # Build result
        optimal_vars = self._array_to_dict(result.x)
        final_outputs = eval_func(optimal_vars)

        # Check final constraint satisfaction
        all_satisfied = all(
            constraint.is_satisfied(final_outputs)
            for constraint in self.constraints.values()
        )

        optimal_obj = final_outputs.get(self.objective_name, result.fun)
        if not self.objective_minimize:
            optimal_obj = -result.fun if hasattr(result, 'fun') else optimal_obj

        return OptimizationResult(
            success=result.success,
            optimal_variables=optimal_vars,
            optimal_objective=optimal_obj,
            optimal_outputs=final_outputs,
            history=self.history,
            message=result.message if hasattr(result, 'message') else str(result),
            n_iterations=self._iteration_count,
            n_evaluations=self._eval_count,
            constraints_satisfied=all_satisfied
        )

    def reset(self):
        """Reset optimizer state for new run."""
        self.history = OptimizationHistory()
        self._iteration_count = 0
        self._eval_count = 0


# =============================================================================
# MULTI-POINT OPTIMIZER (Pareto)
# =============================================================================

class MultiPointOptimizer:
    """
    Multi-point Pareto optimization for throttle envelope design.

    Optimizes engine design across multiple operating points simultaneously,
    finding the Pareto front of non-dominated designs.

    Example:
        >>> optimizer = MultiPointOptimizer()
        >>> optimizer.add_operating_point('full', weight=1.0, pc_range=(40, 50), mr_range=(2.0, 2.5))
        >>> optimizer.add_operating_point('throttle', weight=0.8, pc_range=(10, 20), mr_range=(1.8, 2.8))
        >>> optimizer.set_objectives(['max_thrust', 'min_wall_temp', 'min_pressure_drop'])
        >>> result = optimizer.optimize(eval_engine)
    """

    def __init__(
        self,
        n_population: int = 50,
        n_generations: int = 100
    ):
        """
        Initialize multi-point optimizer.

        Args:
            n_population: Population size for evolutionary algorithm
            n_generations: Number of generations
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for optimization")

        self.n_population = n_population
        self.n_generations = n_generations
        self.operating_points: List[OperatingPoint] = []
        self.objectives: Dict[str, bool] = {}  # name -> minimize
        self.shared_variables: Dict[str, DesignVariable] = {}
        self._all_points: List[ParetoPoint] = []

    def add_operating_point(
        self,
        name: str,
        weight: float,
        pc_range: Tuple[float, float],
        mr_range: Tuple[float, float],
        constraints: Optional[List[Constraint]] = None
    ) -> 'MultiPointOptimizer':
        """
        Add an operating point for multi-point optimization.

        Args:
            name: Operating point identifier
            weight: Importance weight (higher = more important)
            pc_range: Chamber pressure range [min, max] bar
            mr_range: Mixture ratio range [min, max]
            constraints: Optional constraints specific to this point

        Returns:
            Self for method chaining
        """
        self.operating_points.append(OperatingPoint(
            name=name,
            weight=weight,
            pc_range=pc_range,
            mr_range=mr_range,
            constraints=constraints or []
        ))
        return self

    def add_shared_variable(
        self,
        name: str,
        min_val: float,
        max_val: float,
        initial: float
    ) -> 'MultiPointOptimizer':
        """
        Add a design variable shared across all operating points.

        These are geometry parameters like channel height that remain
        constant while pc/mr vary.

        Args:
            name: Variable name
            min_val: Lower bound
            max_val: Upper bound
            initial: Starting value

        Returns:
            Self for method chaining
        """
        self.shared_variables[name] = DesignVariable(
            name=name,
            min_val=min_val,
            max_val=max_val,
            initial=initial
        )
        return self

    def set_objectives(
        self,
        objectives: List[str]
    ) -> 'MultiPointOptimizer':
        """
        Set optimization objectives.

        Objective names should follow convention:
        - 'max_*': Maximize (e.g., 'max_thrust', 'max_isp')
        - 'min_*': Minimize (e.g., 'min_wall_temp', 'min_pressure_drop')

        Args:
            objectives: List of objective names

        Returns:
            Self for method chaining
        """
        self.objectives.clear()
        for obj in objectives:
            # Parse direction from name
            if obj.startswith('max_'):
                self.objectives[obj] = False  # maximize
            elif obj.startswith('min_'):
                self.objectives[obj] = True   # minimize
            else:
                # Default to minimize
                self.objectives[obj] = True
                warnings.warn(f"Objective '{obj}' has no direction prefix. Defaulting to minimize.")

        return self

    def _compute_weighted_objectives(
        self,
        eval_func: Callable,
        shared_vars: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Evaluate weighted objectives across all operating points.

        Returns aggregate objectives considering operating point weights.
        """
        aggregated = {obj: 0.0 for obj in self.objectives}
        total_weight = sum(op.weight for op in self.operating_points)

        for op in self.operating_points:
            # Sample from operating point range
            pc = (op.pc_range[0] + op.pc_range[1]) / 2
            mr = (op.mr_range[0] + op.mr_range[1]) / 2

            # Build full variable set
            variables = {**shared_vars, 'pc': pc, 'mr': mr}

            # Evaluate
            outputs = eval_func(variables)

            # Aggregate objectives with weight
            normalized_weight = op.weight / total_weight
            for obj_name in self.objectives:
                # Strip prefix for lookup
                output_key = obj_name.replace('max_', '').replace('min_', '')
                if output_key in outputs:
                    aggregated[obj_name] += outputs[output_key] * normalized_weight

        return aggregated

    def _is_dominated(self, point: ParetoPoint, other: ParetoPoint) -> bool:
        """Check if 'point' is dominated by 'other'."""
        dominated = False
        for obj_name, minimize in self.objectives.items():
            p_val = point.objectives[obj_name]
            o_val = other.objectives[obj_name]

            if minimize:
                if p_val < o_val:
                    return False
                if p_val > o_val:
                    dominated = True
            else:
                if p_val > o_val:
                    return False
                if p_val < o_val:
                    dominated = True

        return dominated

    def _compute_pareto_front(self, points: List[ParetoPoint]) -> List[ParetoPoint]:
        """Extract non-dominated points from a set."""
        pareto_front = []

        for point in points:
            is_dominated = False
            for other in points:
                if point is not other and self._is_dominated(point, other):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(point)

        return pareto_front

    def optimize(
        self,
        eval_func: Callable[[Dict[str, float]], Dict[str, float]],
        verbose: bool = False
    ) -> ParetoResult:
        """
        Run multi-point Pareto optimization.

        Uses weighted-sum scalarization with varying weights to approximate
        the Pareto front.

        Args:
            eval_func: Evaluation function taking variables dict, returning outputs dict
            verbose: Print progress

        Returns:
            ParetoResult with Pareto front and all evaluated points
        """
        if len(self.objectives) == 0:
            raise ValueError("No objectives set. Call set_objectives() first.")

        if len(self.operating_points) == 0:
            raise ValueError("No operating points defined. Call add_operating_point() first.")

        self._all_points = []
        histories: Dict[str, OptimizationHistory] = {
            obj: OptimizationHistory() for obj in self.objectives
        }

        # Generate weight combinations for scalarization
        n_weights = self.n_population
        weight_sets = self._generate_weight_sets(len(self.objectives), n_weights)

        if verbose:
            print(f"Running Pareto optimization with {len(weight_sets)} weight combinations...")

        # For each weight set, optimize the weighted sum
        for i, weights in enumerate(weight_sets):
            if verbose and i % 10 == 0:
                print(f"  Weight set {i+1}/{len(weight_sets)}")

            # Create weighted objective function
            def weighted_objective(x: np.ndarray) -> float:
                # Build variable dict from shared variables
                var_names = list(self.shared_variables.keys())
                shared_vars = {name: x[j] for j, name in enumerate(var_names)}

                # Compute objectives across operating points
                obj_values = self._compute_weighted_objectives(eval_func, shared_vars)

                # Weighted sum
                weighted_sum = 0.0
                for j, (obj_name, minimize) in enumerate(self.objectives.items()):
                    val = obj_values[obj_name]
                    if not minimize:
                        val = -val  # Convert to minimization
                    weighted_sum += weights[j] * val

                return weighted_sum

            # Run optimization
            if len(self.shared_variables) > 0:
                bounds = [var.bounds for var in self.shared_variables.values()]
                x0 = np.array([var.initial for var in self.shared_variables.values()])

                result = minimize(
                    weighted_objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': self.n_generations}
                )

                # Evaluate at optimal point
                var_names = list(self.shared_variables.keys())
                shared_vars = {name: result.x[j] for j, name in enumerate(var_names)}
            else:
                shared_vars = {}

            # Evaluate full objectives
            obj_values = self._compute_weighted_objectives(eval_func, shared_vars)

            # Also get full outputs at center point of first operating point
            op = self.operating_points[0]
            pc = (op.pc_range[0] + op.pc_range[1]) / 2
            mr = (op.mr_range[0] + op.mr_range[1]) / 2
            full_vars = {**shared_vars, 'pc': pc, 'mr': mr}
            full_outputs = eval_func(full_vars)

            # Record point
            pareto_point = ParetoPoint(
                objectives=obj_values,
                variables=full_vars,
                outputs=full_outputs,
                operating_point=op.name
            )
            self._all_points.append(pareto_point)

            # Record in histories
            for obj_name in self.objectives:
                histories[obj_name].add_iteration(
                    i, obj_values[obj_name], full_vars, {}, True
                )

        # Extract Pareto front
        pareto_front = self._compute_pareto_front(self._all_points)

        if verbose:
            print(f"Found {len(pareto_front)} Pareto-optimal points")

        return ParetoResult(
            pareto_front=pareto_front,
            all_points=self._all_points,
            operating_points=self.operating_points,
            objectives=self.objectives,
            history=histories
        )

    def _generate_weight_sets(self, n_objectives: int, n_sets: int) -> List[List[float]]:
        """
        Generate weight combinations for weighted-sum scalarization.

        Uses uniform spacing on simplex for even coverage.
        """
        if n_objectives == 1:
            return [[1.0]]

        if n_objectives == 2:
            # Simple linear interpolation
            weights = []
            for i in range(n_sets):
                w1 = i / (n_sets - 1)
                weights.append([w1, 1 - w1])
            return weights

        # For 3+ objectives, use random sampling on simplex
        np.random.seed(42)
        weights = []
        for _ in range(n_sets):
            raw = np.random.random(n_objectives)
            normalized = raw / raw.sum()
            weights.append(normalized.tolist())

        return weights

    def sample_operating_envelope(
        self,
        eval_func: Callable,
        n_samples_per_point: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Sample the operating envelope to understand design space.

        Args:
            eval_func: Evaluation function
            n_samples_per_point: Samples per operating point

        Returns:
            List of sample results with variables and outputs
        """
        samples = []

        for op in self.operating_points:
            # Latin hypercube sampling within operating point bounds
            np.random.seed(42)

            for _ in range(n_samples_per_point):
                pc = np.random.uniform(op.pc_range[0], op.pc_range[1])
                mr = np.random.uniform(op.mr_range[0], op.mr_range[1])

                # Add shared variables at their initial values
                variables = {'pc': pc, 'mr': mr}
                for name, var in self.shared_variables.items():
                    variables[name] = var.initial

                # Evaluate
                outputs = eval_func(variables)

                samples.append({
                    'operating_point': op.name,
                    'variables': variables,
                    'outputs': outputs
                })

        return samples
