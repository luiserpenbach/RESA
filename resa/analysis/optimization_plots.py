"""
Optimization Visualization Module using Plotly.

Provides interactive visualizations for optimization analysis:
- ConvergencePlotter: Objective function convergence over iterations
- ParetoFrontPlotter: Multi-objective Pareto front visualization
- DesignSpacePlotter: Design variable space exploration

All plots use Plotly for interactive, web-ready visualizations.

Example:
    >>> from resa.analysis.optimization import ThrottleOptimizer, OptimizationResult
    >>> from resa.analysis.optimization_plots import ConvergencePlotter
    >>>
    >>> # After running optimization...
    >>> plotter = ConvergencePlotter()
    >>> fig = plotter.create_figure(result)
    >>> fig.show()
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from resa.visualization.themes import PlotTheme, DEFAULT_THEME

if TYPE_CHECKING:
    from resa.analysis.optimization import (
        OptimizationResult,
        ParetoResult,
        OptimizationHistory,
        DesignVariable,
    )


# =============================================================================
# CONVERGENCE PLOTTER
# =============================================================================

class ConvergencePlotter:
    """
    Visualizes optimization convergence history.

    Shows:
    - Objective function value vs iteration
    - Constraint violation over time
    - Feasibility markers
    - Best-so-far trajectory

    Example:
        >>> plotter = ConvergencePlotter()
        >>> fig = plotter.create_figure(optimization_result)
        >>> fig.show()
    """

    def __init__(self, theme: Optional[PlotTheme] = None):
        """Initialize with optional custom theme."""
        self.theme = theme or DEFAULT_THEME

    def create_figure(
        self,
        result: 'OptimizationResult',
        show_constraints: bool = True,
        show_best_so_far: bool = True,
        title: str = "Optimization Convergence"
    ) -> go.Figure:
        """
        Create convergence plot from optimization result.

        Args:
            result: OptimizationResult from optimization run
            show_constraints: Show constraint violation subplot
            show_best_so_far: Overlay best-so-far trajectory
            title: Plot title

        Returns:
            Plotly Figure with convergence visualization
        """
        history = result.history

        if len(history.iterations) == 0:
            # Empty history - create placeholder
            fig = go.Figure()
            fig.add_annotation(
                text="No optimization history available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig

        # Determine subplot layout
        n_rows = 2 if show_constraints and history.constraint_violations else 1

        fig = make_subplots(
            rows=n_rows, cols=1,
            subplot_titles=(
                ["Objective Value", "Constraint Violations"][:n_rows]
            ),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4][:n_rows] if n_rows > 1 else None
        )

        iterations = history.iterations
        objectives = history.objective_values
        feasible = history.feasible

        # ========== Main Objective Plot ==========

        # Separate feasible/infeasible points
        feas_iters = [it for it, f in zip(iterations, feasible) if f]
        feas_objs = [obj for obj, f in zip(objectives, feasible) if f]
        infeas_iters = [it for it, f in zip(iterations, feasible) if not f]
        infeas_objs = [obj for obj, f in zip(objectives, feasible) if not f]

        # Plot infeasible points
        if infeas_iters:
            fig.add_trace(
                go.Scatter(
                    x=infeas_iters,
                    y=infeas_objs,
                    mode='markers',
                    name='Infeasible',
                    marker=dict(
                        color='rgba(214, 39, 40, 0.5)',
                        size=8,
                        symbol='x',
                        line=dict(width=1, color=self.theme.danger)
                    ),
                    hovertemplate="Iter: %{x}<br>Objective: %{y:.4g}<br>(Infeasible)<extra></extra>"
                ),
                row=1, col=1
            )

        # Plot feasible points
        if feas_iters:
            fig.add_trace(
                go.Scatter(
                    x=feas_iters,
                    y=feas_objs,
                    mode='markers',
                    name='Feasible',
                    marker=dict(
                        color=self.theme.primary,
                        size=8,
                        symbol='circle',
                    ),
                    hovertemplate="Iter: %{x}<br>Objective: %{y:.4g}<br>(Feasible)<extra></extra>"
                ),
                row=1, col=1
            )

        # Best-so-far trajectory
        if show_best_so_far and feas_objs:
            best_so_far = []
            best = float('inf')
            for obj, feas in zip(objectives, feasible):
                if feas and obj < best:
                    best = obj
                best_so_far.append(best if best != float('inf') else None)

            # Filter out None values
            valid_bsf = [(it, bsf) for it, bsf in zip(iterations, best_so_far) if bsf is not None]
            if valid_bsf:
                bsf_iters, bsf_vals = zip(*valid_bsf)
                fig.add_trace(
                    go.Scatter(
                        x=bsf_iters,
                        y=bsf_vals,
                        mode='lines',
                        name='Best So Far',
                        line=dict(
                            color=self.theme.accent,
                            width=2,
                            dash='dash'
                        ),
                        hovertemplate="Iter: %{x}<br>Best: %{y:.4g}<extra></extra>"
                    ),
                    row=1, col=1
                )

        # Mark optimal point
        if result.constraints_satisfied:
            fig.add_trace(
                go.Scatter(
                    x=[iterations[-1]],
                    y=[result.optimal_objective],
                    mode='markers',
                    name='Optimal',
                    marker=dict(
                        color='gold',
                        size=15,
                        symbol='star',
                        line=dict(color='black', width=2)
                    ),
                    hovertemplate=f"Optimal: {result.optimal_objective:.4g}<extra></extra>"
                ),
                row=1, col=1
            )

        # ========== Constraint Violation Plot ==========
        if n_rows > 1 and history.constraint_violations:
            constraint_names = list(history.constraint_violations[0].keys())
            colors = self.theme.get_color_sequence()

            for i, const_name in enumerate(constraint_names):
                violations = [cv.get(const_name, 0) for cv in history.constraint_violations]

                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=violations,
                        mode='lines+markers',
                        name=const_name,
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=5),
                        hovertemplate=f"{const_name}<br>Iter: %{{x}}<br>Violation: %{{y:.4g}}<extra></extra>"
                    ),
                    row=2, col=1
                )

            # Zero line (feasibility boundary)
            fig.add_hline(
                y=0, row=2, col=1,
                line=dict(color='gray', dash='dash', width=1),
                annotation_text="Feasible",
                annotation_position="top right"
            )

        # ========== Layout ==========
        fig.update_xaxes(title_text="Iteration", row=1, col=1)
        fig.update_yaxes(title_text="Objective Value", row=1, col=1)

        if n_rows > 1:
            fig.update_xaxes(title_text="Iteration", row=2, col=1)
            fig.update_yaxes(title_text="Constraint Violation", row=2, col=1)

        fig.update_layout(
            height=500 if n_rows == 1 else 700,
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2 if n_rows == 1 else -0.15,
                xanchor="center",
                x=0.5
            )
        )

        self.theme.apply_to_figure(fig)

        return fig

    def create_variable_history(
        self,
        result: 'OptimizationResult',
        variables: Optional[List[str]] = None,
        title: str = "Design Variable History"
    ) -> go.Figure:
        """
        Create plot showing how design variables evolved during optimization.

        Args:
            result: OptimizationResult from optimization run
            variables: Specific variables to show (None = all)
            title: Plot title

        Returns:
            Plotly Figure with variable trajectories
        """
        history = result.history

        if len(history.variable_values) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No variable history available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        all_vars = list(history.variable_values[0].keys())
        vars_to_plot = variables if variables else all_vars

        # Create subplots for each variable
        n_vars = len(vars_to_plot)
        fig = make_subplots(
            rows=n_vars, cols=1,
            subplot_titles=vars_to_plot,
            vertical_spacing=0.1
        )

        iterations = history.iterations
        colors = self.theme.get_color_sequence()

        for i, var_name in enumerate(vars_to_plot):
            values = [vv.get(var_name, np.nan) for vv in history.variable_values]

            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=values,
                    mode='lines+markers',
                    name=var_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate=f"{var_name}: %{{y:.4g}}<br>Iter: %{{x}}<extra></extra>"
                ),
                row=i+1, col=1
            )

            # Mark optimal value
            if var_name in result.optimal_variables:
                opt_val = result.optimal_variables[var_name]
                fig.add_hline(
                    y=opt_val, row=i+1, col=1,
                    line=dict(color='gold', dash='dot', width=2),
                    annotation_text=f"Optimal: {opt_val:.4g}",
                    annotation_position="right"
                )

            fig.update_yaxes(title_text=var_name, row=i+1, col=1)

        fig.update_xaxes(title_text="Iteration", row=n_vars, col=1)

        fig.update_layout(
            height=200 * n_vars + 100,
            title=dict(text=title, x=0.5, font=dict(size=18)),
            showlegend=False
        )

        self.theme.apply_to_figure(fig)

        return fig

    def to_html(
        self,
        fig: go.Figure,
        include_plotlyjs: str = 'cdn',
        full_html: bool = False
    ) -> str:
        """Export figure to embeddable HTML."""
        return fig.to_html(
            include_plotlyjs=include_plotlyjs,
            full_html=full_html
        )


# =============================================================================
# PARETO FRONT PLOTTER
# =============================================================================

class ParetoFrontPlotter:
    """
    Visualizes multi-objective Pareto fronts.

    Shows:
    - 2D Pareto front for bi-objective problems
    - 3D Pareto front for tri-objective problems
    - Parallel coordinates for high-dimensional objectives
    - Trade-off analysis

    Example:
        >>> plotter = ParetoFrontPlotter()
        >>> fig = plotter.create_figure(pareto_result)
        >>> fig.show()
    """

    def __init__(self, theme: Optional[PlotTheme] = None):
        """Initialize with optional custom theme."""
        self.theme = theme or DEFAULT_THEME

    def create_figure(
        self,
        result: 'ParetoResult',
        show_all_points: bool = True,
        show_utopia: bool = True,
        title: str = "Pareto Front"
    ) -> go.Figure:
        """
        Create Pareto front visualization.

        Automatically selects 2D, 3D, or parallel coordinates based on
        number of objectives.

        Args:
            result: ParetoResult from multi-point optimization
            show_all_points: Show all evaluated points (not just Pareto front)
            show_utopia: Show utopia point (best of each objective)
            title: Plot title

        Returns:
            Plotly Figure with Pareto front
        """
        n_objectives = len(result.objectives)

        if n_objectives == 2:
            return self._create_2d_pareto(result, show_all_points, show_utopia, title)
        elif n_objectives == 3:
            return self._create_3d_pareto(result, show_all_points, show_utopia, title)
        else:
            return self._create_parallel_coordinates(result, title)

    def _create_2d_pareto(
        self,
        result: 'ParetoResult',
        show_all_points: bool,
        show_utopia: bool,
        title: str
    ) -> go.Figure:
        """Create 2D Pareto front plot."""
        fig = go.Figure()

        obj_names = list(result.objectives.keys())
        obj1, obj2 = obj_names[0], obj_names[1]

        # All evaluated points
        if show_all_points:
            all_x = [p.objectives[obj1] for p in result.all_points]
            all_y = [p.objectives[obj2] for p in result.all_points]

            fig.add_trace(go.Scatter(
                x=all_x,
                y=all_y,
                mode='markers',
                name='Evaluated Points',
                marker=dict(
                    color='rgba(150, 150, 150, 0.5)',
                    size=8,
                    symbol='circle'
                ),
                hovertemplate=f"{obj1}: %{{x:.4g}}<br>{obj2}: %{{y:.4g}}<extra></extra>"
            ))

        # Pareto front points
        pareto_x = [p.objectives[obj1] for p in result.pareto_front]
        pareto_y = [p.objectives[obj2] for p in result.pareto_front]

        # Sort for connected line
        sorted_indices = np.argsort(pareto_x)
        sorted_x = [pareto_x[i] for i in sorted_indices]
        sorted_y = [pareto_y[i] for i in sorted_indices]

        # Pareto front line
        fig.add_trace(go.Scatter(
            x=sorted_x,
            y=sorted_y,
            mode='lines',
            name='Pareto Front',
            line=dict(color=self.theme.primary, width=3),
            hoverinfo='skip'
        ))

        # Pareto front points
        fig.add_trace(go.Scatter(
            x=pareto_x,
            y=pareto_y,
            mode='markers',
            name='Pareto Optimal',
            marker=dict(
                color=self.theme.secondary,
                size=12,
                symbol='diamond',
                line=dict(color='white', width=1)
            ),
            hovertemplate=f"{obj1}: %{{x:.4g}}<br>{obj2}: %{{y:.4g}}<extra>Pareto Optimal</extra>"
        ))

        # Utopia point
        if show_utopia:
            utopia_x = min(pareto_x) if result.objectives[obj1] else max(pareto_x)
            utopia_y = min(pareto_y) if result.objectives[obj2] else max(pareto_y)

            fig.add_trace(go.Scatter(
                x=[utopia_x],
                y=[utopia_y],
                mode='markers',
                name='Utopia Point',
                marker=dict(
                    color='gold',
                    size=15,
                    symbol='star',
                    line=dict(color='black', width=2)
                ),
                hovertemplate=f"Utopia<br>{obj1}: {utopia_x:.4g}<br>{obj2}: {utopia_y:.4g}<extra></extra>"
            ))

        # Axis labels with direction indicators
        x_label = f"{obj1} {'(minimize)' if result.objectives[obj1] else '(maximize)'}"
        y_label = f"{obj2} {'(minimize)' if result.objectives[obj2] else '(maximize)'}"

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=600,
            width=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )

        self.theme.apply_to_figure(fig)

        return fig

    def _create_3d_pareto(
        self,
        result: 'ParetoResult',
        show_all_points: bool,
        show_utopia: bool,
        title: str
    ) -> go.Figure:
        """Create 3D Pareto front plot."""
        fig = go.Figure()

        obj_names = list(result.objectives.keys())
        obj1, obj2, obj3 = obj_names[0], obj_names[1], obj_names[2]

        # All evaluated points
        if show_all_points:
            all_x = [p.objectives[obj1] for p in result.all_points]
            all_y = [p.objectives[obj2] for p in result.all_points]
            all_z = [p.objectives[obj3] for p in result.all_points]

            fig.add_trace(go.Scatter3d(
                x=all_x, y=all_y, z=all_z,
                mode='markers',
                name='Evaluated Points',
                marker=dict(
                    color='rgba(150, 150, 150, 0.4)',
                    size=4,
                    symbol='circle'
                ),
                hovertemplate=f"{obj1}: %{{x:.4g}}<br>{obj2}: %{{y:.4g}}<br>{obj3}: %{{z:.4g}}<extra></extra>"
            ))

        # Pareto front points
        pareto_x = [p.objectives[obj1] for p in result.pareto_front]
        pareto_y = [p.objectives[obj2] for p in result.pareto_front]
        pareto_z = [p.objectives[obj3] for p in result.pareto_front]

        fig.add_trace(go.Scatter3d(
            x=pareto_x, y=pareto_y, z=pareto_z,
            mode='markers',
            name='Pareto Optimal',
            marker=dict(
                color=self.theme.secondary,
                size=8,
                symbol='diamond',
                line=dict(color='white', width=1)
            ),
            hovertemplate=f"{obj1}: %{{x:.4g}}<br>{obj2}: %{{y:.4g}}<br>{obj3}: %{{z:.4g}}<extra>Pareto</extra>"
        ))

        # Utopia point
        if show_utopia:
            utopia = [
                min(pareto_x) if result.objectives[obj1] else max(pareto_x),
                min(pareto_y) if result.objectives[obj2] else max(pareto_y),
                min(pareto_z) if result.objectives[obj3] else max(pareto_z),
            ]

            fig.add_trace(go.Scatter3d(
                x=[utopia[0]], y=[utopia[1]], z=[utopia[2]],
                mode='markers',
                name='Utopia Point',
                marker=dict(
                    color='gold',
                    size=12,
                    symbol='diamond',
                    line=dict(color='black', width=2)
                )
            ))

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            scene=dict(
                xaxis_title=obj1,
                yaxis_title=obj2,
                zaxis_title=obj3,
            ),
            height=700,
            width=900,
            showlegend=True
        )

        return fig

    def _create_parallel_coordinates(
        self,
        result: 'ParetoResult',
        title: str
    ) -> go.Figure:
        """Create parallel coordinates plot for high-dimensional Pareto fronts."""
        obj_names = list(result.objectives.keys())

        # Build dimensions
        dimensions = []
        for obj_name in obj_names:
            values = [p.objectives[obj_name] for p in result.pareto_front]
            minimize = result.objectives[obj_name]

            dimensions.append(dict(
                label=obj_name,
                values=values,
                range=[min(values), max(values)]
            ))

        # Color by first objective
        color_values = [p.objectives[obj_names[0]] for p in result.pareto_front]

        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=color_values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=obj_names[0])
                ),
                dimensions=dimensions
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=500,
            width=900
        )

        return fig

    def create_trade_off_table(
        self,
        result: 'ParetoResult',
        n_points: int = 5
    ) -> go.Figure:
        """
        Create a table showing trade-offs between Pareto points.

        Args:
            result: ParetoResult from optimization
            n_points: Number of representative points to show

        Returns:
            Plotly Figure with trade-off table
        """
        # Select representative points
        if len(result.pareto_front) <= n_points:
            points = result.pareto_front
        else:
            # Evenly sample from Pareto front
            indices = np.linspace(0, len(result.pareto_front) - 1, n_points, dtype=int)
            points = [result.pareto_front[i] for i in indices]

        # Build table data
        obj_names = list(result.objectives.keys())
        var_names = list(points[0].variables.keys()) if points else []

        headers = ['Point'] + obj_names + var_names
        cells = [[] for _ in headers]

        for i, point in enumerate(points):
            cells[0].append(f"P{i+1}")
            for j, obj_name in enumerate(obj_names):
                cells[j+1].append(f"{point.objectives[obj_name]:.4g}")
            for k, var_name in enumerate(var_names):
                cells[len(obj_names) + 1 + k].append(f"{point.variables.get(var_name, 'N/A'):.4g}")

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                fill_color=self.theme.primary,
                font=dict(color='white', size=12),
                align='center'
            ),
            cells=dict(
                values=cells,
                fill_color='white',
                align='center',
                font=dict(size=11)
            )
        )])

        fig.update_layout(
            title=dict(text="Pareto Trade-off Summary", x=0.5),
            height=300 + 30 * len(points)
        )

        return fig

    def to_html(
        self,
        fig: go.Figure,
        include_plotlyjs: str = 'cdn',
        full_html: bool = False
    ) -> str:
        """Export figure to embeddable HTML."""
        return fig.to_html(
            include_plotlyjs=include_plotlyjs,
            full_html=full_html
        )


# =============================================================================
# DESIGN SPACE PLOTTER
# =============================================================================

class DesignSpacePlotter:
    """
    Visualizes design variable space exploration.

    Shows:
    - 2D scatter of variable combinations with objective coloring
    - Contour plots of objective over design space
    - Constraint boundaries
    - Optimal point location

    Example:
        >>> plotter = DesignSpacePlotter()
        >>> fig = plotter.create_figure(result, x_var='pc', y_var='mr')
        >>> fig.show()
    """

    def __init__(self, theme: Optional[PlotTheme] = None):
        """Initialize with optional custom theme."""
        self.theme = theme or DEFAULT_THEME

    def create_figure(
        self,
        result: 'OptimizationResult',
        x_var: str,
        y_var: str,
        color_by: Optional[str] = None,
        show_optimal: bool = True,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create design space scatter plot.

        Args:
            result: OptimizationResult from optimization
            x_var: Variable name for x-axis
            y_var: Variable name for y-axis
            color_by: Variable or output to color points by (None = feasibility)
            show_optimal: Highlight optimal point
            title: Plot title

        Returns:
            Plotly Figure with design space visualization
        """
        history = result.history

        if len(history.variable_values) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No design space data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Extract data
        x_vals = [vv.get(x_var) for vv in history.variable_values]
        y_vals = [vv.get(y_var) for vv in history.variable_values]

        if None in x_vals or None in y_vals:
            raise ValueError(f"Variables '{x_var}' or '{y_var}' not found in history")

        fig = go.Figure()

        # Determine coloring
        if color_by is None:
            # Color by feasibility
            colors = ['green' if f else 'red' for f in history.feasible]
            colorbar_title = "Feasible"
            colorscale = None
        else:
            # Color by objective or output
            colors = history.objective_values if color_by == 'objective' else [
                vv.get(color_by, 0) for vv in history.variable_values
            ]
            colorbar_title = color_by
            colorscale = 'Viridis'

        # Scatter plot
        if colorscale:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(
                    color=colors,
                    colorscale=colorscale,
                    size=10,
                    colorbar=dict(title=colorbar_title),
                    line=dict(color='white', width=0.5)
                ),
                name='Evaluated Points',
                hovertemplate=f"{x_var}: %{{x:.4g}}<br>{y_var}: %{{y:.4g}}<br>{colorbar_title}: %{{marker.color:.4g}}<extra></extra>"
            ))
        else:
            # Separate by feasibility
            feas_x = [x for x, f in zip(x_vals, history.feasible) if f]
            feas_y = [y for y, f in zip(y_vals, history.feasible) if f]
            infeas_x = [x for x, f in zip(x_vals, history.feasible) if not f]
            infeas_y = [y for y, f in zip(y_vals, history.feasible) if not f]

            if feas_x:
                fig.add_trace(go.Scatter(
                    x=feas_x, y=feas_y,
                    mode='markers',
                    name='Feasible',
                    marker=dict(color=self.theme.accent, size=10),
                    hovertemplate=f"{x_var}: %{{x:.4g}}<br>{y_var}: %{{y:.4g}}<extra>Feasible</extra>"
                ))

            if infeas_x:
                fig.add_trace(go.Scatter(
                    x=infeas_x, y=infeas_y,
                    mode='markers',
                    name='Infeasible',
                    marker=dict(color=self.theme.danger, size=8, symbol='x'),
                    hovertemplate=f"{x_var}: %{{x:.4g}}<br>{y_var}: %{{y:.4g}}<extra>Infeasible</extra>"
                ))

        # Optimal point
        if show_optimal and x_var in result.optimal_variables and y_var in result.optimal_variables:
            opt_x = result.optimal_variables[x_var]
            opt_y = result.optimal_variables[y_var]

            fig.add_trace(go.Scatter(
                x=[opt_x],
                y=[opt_y],
                mode='markers',
                name='Optimal',
                marker=dict(
                    color='gold',
                    size=18,
                    symbol='star',
                    line=dict(color='black', width=2)
                ),
                hovertemplate=f"Optimal<br>{x_var}: {opt_x:.4g}<br>{y_var}: {opt_y:.4g}<extra></extra>"
            ))

        # Layout
        plot_title = title or f"Design Space: {x_var} vs {y_var}"
        fig.update_layout(
            title=dict(text=plot_title, x=0.5, font=dict(size=18)),
            xaxis_title=x_var,
            yaxis_title=y_var,
            height=600,
            width=700,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        self.theme.apply_to_figure(fig)

        return fig

    def create_contour(
        self,
        eval_func: Any,
        x_var: 'DesignVariable',
        y_var: 'DesignVariable',
        objective: str,
        n_grid: int = 30,
        constraints: Optional[List[Any]] = None,
        optimal_point: Optional[Dict[str, float]] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create contour plot of objective over design space.

        Args:
            eval_func: Evaluation function
            x_var: X-axis design variable
            y_var: Y-axis design variable
            objective: Output variable to contour
            n_grid: Grid resolution
            constraints: Optional constraints to show as boundaries
            optimal_point: Optional optimal point to highlight
            title: Plot title

        Returns:
            Plotly Figure with contour visualization
        """
        # Create grid
        x_range = np.linspace(x_var.min_val, x_var.max_val, n_grid)
        y_range = np.linspace(y_var.min_val, y_var.max_val, n_grid)
        X, Y = np.meshgrid(x_range, y_range)

        # Evaluate on grid
        Z = np.zeros_like(X)
        for i in range(n_grid):
            for j in range(n_grid):
                variables = {x_var.name: X[i, j], y_var.name: Y[i, j]}
                outputs = eval_func(variables)
                Z[i, j] = outputs.get(objective, np.nan)

        fig = go.Figure()

        # Contour plot
        fig.add_trace(go.Contour(
            x=x_range,
            y=y_range,
            z=Z,
            colorscale='Viridis',
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            colorbar=dict(title=objective),
            hovertemplate=f"{x_var.name}: %{{x:.4g}}<br>{y_var.name}: %{{y:.4g}}<br>{objective}: %{{z:.4g}}<extra></extra>"
        ))

        # Constraint boundaries
        if constraints:
            for constraint in constraints:
                # Evaluate constraint on grid
                C = np.zeros_like(X)
                for i in range(n_grid):
                    for j in range(n_grid):
                        variables = {x_var.name: X[i, j], y_var.name: Y[i, j]}
                        outputs = eval_func(variables)
                        C[i, j] = constraint.evaluate(outputs)

                # Add constraint boundary (where violation = 0)
                fig.add_trace(go.Contour(
                    x=x_range,
                    y=y_range,
                    z=C,
                    contours=dict(
                        start=0, end=0, size=1,
                        coloring='none',
                        showlabels=True
                    ),
                    line=dict(color='red', width=3, dash='dash'),
                    showscale=False,
                    name=f"Constraint: {constraint.name}",
                    hoverinfo='skip'
                ))

        # Optimal point
        if optimal_point:
            opt_x = optimal_point.get(x_var.name)
            opt_y = optimal_point.get(y_var.name)
            if opt_x is not None and opt_y is not None:
                fig.add_trace(go.Scatter(
                    x=[opt_x],
                    y=[opt_y],
                    mode='markers',
                    name='Optimal',
                    marker=dict(
                        color='gold',
                        size=18,
                        symbol='star',
                        line=dict(color='black', width=2)
                    )
                ))

        # Layout
        plot_title = title or f"{objective} Contour"
        fig.update_layout(
            title=dict(text=plot_title, x=0.5, font=dict(size=18)),
            xaxis_title=x_var.name,
            yaxis_title=y_var.name,
            height=600,
            width=700,
            showlegend=True
        )

        self.theme.apply_to_figure(fig)

        return fig

    def create_sensitivity_bars(
        self,
        result: 'OptimizationResult',
        sensitivities: Optional[Dict[str, float]] = None,
        title: str = "Variable Sensitivity"
    ) -> go.Figure:
        """
        Create bar chart showing sensitivity of objective to each variable.

        Args:
            result: OptimizationResult (used for variable names if sensitivities not provided)
            sensitivities: Dict mapping variable names to sensitivity values.
                          If None, estimates from optimization history.
            title: Plot title

        Returns:
            Plotly Figure with sensitivity bar chart
        """
        if sensitivities is None:
            # Estimate sensitivities from history
            sensitivities = self._estimate_sensitivities(result)

        var_names = list(sensitivities.keys())
        values = list(sensitivities.values())

        # Normalize to percentages
        total = sum(abs(v) for v in values) or 1
        percentages = [100 * abs(v) / total for v in values]

        # Colors based on sign
        colors = [self.theme.accent if v >= 0 else self.theme.danger for v in values]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=var_names,
            y=percentages,
            marker_color=colors,
            text=[f"{p:.1f}%" for p in percentages],
            textposition='outside',
            hovertemplate="%{x}<br>Sensitivity: %{y:.1f}%<extra></extra>"
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            xaxis_title="Design Variable",
            yaxis_title="Relative Sensitivity [%]",
            height=400,
            width=600,
            showlegend=False
        )

        self.theme.apply_to_figure(fig)

        return fig

    def _estimate_sensitivities(
        self,
        result: 'OptimizationResult'
    ) -> Dict[str, float]:
        """Estimate sensitivities from optimization history using variance."""
        history = result.history

        if len(history.variable_values) < 2:
            return {}

        sensitivities = {}
        var_names = list(history.variable_values[0].keys())
        objectives = history.objective_values

        for var_name in var_names:
            values = [vv.get(var_name, 0) for vv in history.variable_values]

            # Simple correlation-based sensitivity
            if np.std(values) > 0 and np.std(objectives) > 0:
                correlation = np.corrcoef(values, objectives)[0, 1]
                sensitivities[var_name] = correlation if not np.isnan(correlation) else 0
            else:
                sensitivities[var_name] = 0

        return sensitivities

    def to_html(
        self,
        fig: go.Figure,
        include_plotlyjs: str = 'cdn',
        full_html: bool = False
    ) -> str:
        """Export figure to embeddable HTML."""
        return fig.to_html(
            include_plotlyjs=include_plotlyjs,
            full_html=full_html
        )
