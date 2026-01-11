"""
Monte Carlo Visualization Tools
===============================

Plotly-based interactive visualizations for Monte Carlo analysis results:
- HistogramPlotter: Output distributions with percentile markers
- ScatterMatrixPlotter: Parameter vs output correlations
- TornadoPlotter: Sensitivity tornado charts
- ConvergencePlotter: Running statistics vs sample count

All plots use Plotly for interactivity and easy HTML embedding.

Example:
    >>> from resa.analysis import MonteCarloAnalysis, HistogramPlotter, TornadoPlotter
    >>>
    >>> mc = MonteCarloAnalysis()
    >>> mc.add_parameter('pc_bar', 25.0, 'normal', std_dev=1.0)
    >>> result = mc.run(1000, engine_func, ['isp', 'thrust'])
    >>>
    >>> hist = HistogramPlotter()
    >>> fig = hist.create_figure(result, 'isp')
    >>> fig.show()
    >>>
    >>> tornado = TornadoPlotter()
    >>> fig = tornado.create_figure(result, 'isp')
    >>> fig.show()
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from scipy import stats

if TYPE_CHECKING:
    from resa.analysis.monte_carlo import MonteCarloResult

# Import theme if available, otherwise use defaults
try:
    from resa.visualization.themes import PlotTheme, EngineeringTheme, DEFAULT_THEME
except ImportError:
    DEFAULT_THEME = None


class BaseMCPlotter:
    """Base class for Monte Carlo plotters with common functionality."""

    # Default color palette for Monte Carlo plots
    DEFAULT_COLORS = {
        'primary': '#2c3e50',
        'secondary': '#e74c3c',
        'accent': '#27ae60',
        'info': '#3498db',
        'warning': '#f39c12',
        'p5_p95': '#e74c3c',
        'p50': '#27ae60',
        'mean': '#3498db',
        'histogram': 'rgba(52, 152, 219, 0.7)',
        'positive_corr': '#27ae60',
        'negative_corr': '#e74c3c',
    }

    def __init__(self, theme: Optional[Any] = None):
        """
        Initialize plotter with optional theme.

        Args:
            theme: PlotTheme instance for consistent styling
        """
        self.theme = theme or DEFAULT_THEME
        self.colors = self.DEFAULT_COLORS.copy()

    def _apply_theme(self, fig: go.Figure) -> go.Figure:
        """Apply theme to figure if available."""
        if self.theme is not None and hasattr(self.theme, 'apply_to_figure'):
            self.theme.apply_to_figure(fig)
        return fig

    def to_html(
        self,
        fig: go.Figure,
        include_plotlyjs: str = 'cdn',
        full_html: bool = False
    ) -> str:
        """
        Export figure to embeddable HTML.

        Args:
            fig: Plotly Figure
            include_plotlyjs: 'cdn', True, False, or path to local plotly.js
            full_html: If True, includes <html> tags

        Returns:
            HTML string
        """
        return fig.to_html(
            include_plotlyjs=include_plotlyjs,
            full_html=full_html
        )

    def show(self, fig: go.Figure) -> None:
        """Display figure interactively."""
        fig.show()


class HistogramPlotter(BaseMCPlotter):
    """
    Creates histogram plots for Monte Carlo output distributions.

    Features:
    - Histogram with KDE overlay
    - P5, P50 (median), P95 percentile markers
    - Mean indicator
    - Normal fit overlay (optional)

    Example:
        >>> plotter = HistogramPlotter()
        >>> fig = plotter.create_figure(result, 'isp')
        >>> fig.show()
    """

    def create_figure(
        self,
        result: 'MonteCarloResult',
        output_name: str,
        n_bins: int = 50,
        show_kde: bool = True,
        show_normal_fit: bool = False,
        show_percentiles: bool = True,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create histogram with statistical overlays.

        Args:
            result: MonteCarloResult from Monte Carlo run
            output_name: Name of output variable to plot
            n_bins: Number of histogram bins
            show_kde: Show kernel density estimate
            show_normal_fit: Show fitted normal distribution
            show_percentiles: Show P5, P50, P95 lines
            title: Custom title

        Returns:
            Plotly Figure
        """
        if output_name not in result.output_samples:
            raise ValueError(f"Unknown output: {output_name}. Available: {result.output_names}")

        samples = result.output_samples[output_name]
        stats_dict = result.statistics.get(output_name, {})

        fig = go.Figure()

        # Compute histogram for proper y-axis scaling
        hist_counts, bin_edges = np.histogram(samples, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        max_density = np.max(hist_counts) * 1.1

        # Main histogram
        fig.add_trace(go.Histogram(
            x=samples,
            nbinsx=n_bins,
            name='Distribution',
            histnorm='probability density',
            marker=dict(
                color=self.colors['histogram'],
                line=dict(color=self.colors['primary'], width=1)
            ),
            hovertemplate='Value: %{x:.4f}<br>Density: %{y:.4f}<extra></extra>'
        ))

        # KDE overlay
        if show_kde:
            kde = stats.gaussian_kde(samples)
            x_kde = np.linspace(np.min(samples), np.max(samples), 200)
            y_kde = kde(x_kde)

            fig.add_trace(go.Scatter(
                x=x_kde,
                y=y_kde,
                mode='lines',
                name='KDE',
                line=dict(color=self.colors['primary'], width=2),
                hovertemplate='Value: %{x:.4f}<br>Density: %{y:.4f}<extra></extra>'
            ))

        # Normal fit overlay
        if show_normal_fit:
            mean = stats_dict.get('mean', np.mean(samples))
            std = stats_dict.get('std', np.std(samples))
            x_norm = np.linspace(np.min(samples), np.max(samples), 200)
            y_norm = stats.norm.pdf(x_norm, mean, std)

            fig.add_trace(go.Scatter(
                x=x_norm,
                y=y_norm,
                mode='lines',
                name='Normal Fit',
                line=dict(color=self.colors['warning'], width=2, dash='dash'),
                hovertemplate='Value: %{x:.4f}<br>Normal PDF: %{y:.4f}<extra></extra>'
            ))

        # Percentile markers
        if show_percentiles:
            p5 = stats_dict.get('P5', np.percentile(samples, 5))
            p50 = stats_dict.get('P50', np.percentile(samples, 50))
            p95 = stats_dict.get('P95', np.percentile(samples, 95))
            mean = stats_dict.get('mean', np.mean(samples))

            # P5 line
            fig.add_vline(
                x=p5,
                line=dict(color=self.colors['p5_p95'], width=2, dash='dash'),
                annotation=dict(
                    text=f"P5: {p5:.4g}",
                    font=dict(color=self.colors['p5_p95']),
                    yshift=10
                )
            )

            # P50 line (median)
            fig.add_vline(
                x=p50,
                line=dict(color=self.colors['p50'], width=2),
                annotation=dict(
                    text=f"P50: {p50:.4g}",
                    font=dict(color=self.colors['p50']),
                    yshift=-10
                )
            )

            # P95 line
            fig.add_vline(
                x=p95,
                line=dict(color=self.colors['p5_p95'], width=2, dash='dash'),
                annotation=dict(
                    text=f"P95: {p95:.4g}",
                    font=dict(color=self.colors['p5_p95']),
                    yshift=10
                )
            )

            # Mean line
            fig.add_vline(
                x=mean,
                line=dict(color=self.colors['mean'], width=2, dash='dot'),
                annotation=dict(
                    text=f"Mean: {mean:.4g}",
                    font=dict(color=self.colors['mean']),
                    yshift=30
                )
            )

        # Layout
        std = stats_dict.get('std', np.std(samples))
        mean = stats_dict.get('mean', np.mean(samples))

        fig.update_layout(
            title=dict(
                text=title or f"Distribution of {output_name}",
                x=0.5,
                font=dict(size=18)
            ),
            xaxis=dict(
                title=output_name,
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(
                title="Probability Density",
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            showlegend=True,
            legend=dict(
                x=0.98,
                y=0.98,
                xanchor='right',
                bgcolor='rgba(255,255,255,0.9)'
            ),
            annotations=[
                dict(
                    text=f"n = {result.n_samples}<br>Mean = {mean:.4g}<br>Std = {std:.4g}",
                    xref="paper", yref="paper",
                    x=0.98, y=0.75,
                    xanchor='right',
                    showarrow=False,
                    font=dict(size=11),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='gray',
                    borderwidth=1,
                    borderpad=5
                )
            ],
            bargap=0.02,
            height=500,
            width=800
        )

        self._apply_theme(fig)
        return fig

    def create_multi_figure(
        self,
        result: 'MonteCarloResult',
        output_names: Optional[List[str]] = None,
        n_cols: int = 2,
        n_bins: int = 40,
        show_percentiles: bool = True
    ) -> go.Figure:
        """
        Create grid of histograms for multiple outputs.

        Args:
            result: MonteCarloResult from Monte Carlo run
            output_names: Outputs to plot (default: all)
            n_cols: Number of columns in grid
            n_bins: Number of histogram bins
            show_percentiles: Show P5/P95 lines

        Returns:
            Plotly Figure with subplots
        """
        if output_names is None:
            output_names = result.output_names

        n_outputs = len(output_names)
        n_rows = int(np.ceil(n_outputs / n_cols))

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=output_names,
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        for i, output_name in enumerate(output_names):
            row = i // n_cols + 1
            col = i % n_cols + 1

            samples = result.output_samples[output_name]
            stats_dict = result.statistics.get(output_name, {})

            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=samples,
                    nbinsx=n_bins,
                    name=output_name,
                    marker=dict(
                        color=self.colors['histogram'],
                        line=dict(color=self.colors['primary'], width=0.5)
                    ),
                    showlegend=False,
                    histnorm='probability density'
                ),
                row=row, col=col
            )

            # Add percentile lines via shapes
            if show_percentiles:
                p5 = stats_dict.get('P5', np.percentile(samples, 5))
                p95 = stats_dict.get('P95', np.percentile(samples, 95))

                # Get subplot reference for shapes
                xref = f"x{i+1}" if i > 0 else "x"
                yref = f"y{i+1}" if i > 0 else "y"

                fig.add_shape(
                    type="line",
                    xref=xref,
                    x0=p5, x1=p5, y0=0, y1=1,
                    yref=f"{yref} domain",
                    line=dict(color=self.colors['p5_p95'], width=2, dash='dash')
                )
                fig.add_shape(
                    type="line",
                    xref=xref,
                    x0=p95, x1=p95, y0=0, y1=1,
                    yref=f"{yref} domain",
                    line=dict(color=self.colors['p5_p95'], width=2, dash='dash')
                )

        fig.update_layout(
            title=dict(
                text="Output Distributions",
                x=0.5,
                font=dict(size=20)
            ),
            height=300 * n_rows,
            width=400 * n_cols,
            showlegend=False
        )

        self._apply_theme(fig)
        return fig


class ScatterMatrixPlotter(BaseMCPlotter):
    """
    Creates scatter matrix plots showing parameter-output correlations.

    Features:
    - Input vs output scatter plots
    - Correlation coefficient display
    - Trend lines (optional)
    - Color-coded by correlation strength

    Example:
        >>> plotter = ScatterMatrixPlotter()
        >>> fig = plotter.create_figure(result)
        >>> fig.show()
    """

    def create_figure(
        self,
        result: 'MonteCarloResult',
        output_name: Optional[str] = None,
        max_points: int = 1000,
        show_trendline: bool = True,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create scatter matrix for one output vs all inputs.

        Args:
            result: MonteCarloResult from Monte Carlo run
            output_name: Output to analyze (default: first)
            max_points: Maximum points to plot (subsampled if exceeded)
            show_trendline: Show linear trend line
            title: Custom title

        Returns:
            Plotly Figure with scatter subplots
        """
        if output_name is None:
            output_name = result.output_names[0]

        if output_name not in result.output_samples:
            raise ValueError(f"Unknown output: {output_name}")

        n_params = len(result.parameter_names)
        n_cols = min(3, n_params)
        n_rows = int(np.ceil(n_params / n_cols))

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=result.parameter_names,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        output_samples = result.output_samples[output_name]
        sensitivity = result.sensitivity.get(output_name, {})
        spearman = sensitivity.get('spearman', {})

        # Subsample if too many points
        n_samples = len(output_samples)
        if n_samples > max_points:
            idx = np.random.choice(n_samples, max_points, replace=False)
            output_plot = output_samples[idx]
        else:
            idx = np.arange(n_samples)
            output_plot = output_samples

        for i, param_name in enumerate(result.parameter_names):
            row = i // n_cols + 1
            col = i % n_cols + 1

            input_samples = result.input_samples[param_name]
            input_plot = input_samples[idx]

            # Get correlation for coloring
            corr = spearman.get(param_name, 0)
            color = self.colors['positive_corr'] if corr >= 0 else self.colors['negative_corr']

            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=input_plot,
                    y=output_plot,
                    mode='markers',
                    name=param_name,
                    marker=dict(
                        size=4,
                        color=color,
                        opacity=0.5
                    ),
                    showlegend=False,
                    hovertemplate=f"{param_name}: %{{x:.4g}}<br>{output_name}: %{{y:.4g}}<extra></extra>"
                ),
                row=row, col=col
            )

            # Trend line
            if show_trendline:
                z = np.polyfit(input_samples, output_samples, 1)
                p = np.poly1d(z)
                x_trend = np.array([np.min(input_samples), np.max(input_samples)])
                y_trend = p(x_trend)

                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode='lines',
                        name='Trend',
                        line=dict(color='black', width=2, dash='dash'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )

            # Add correlation annotation
            fig.add_annotation(
                text=f"r = {corr:.3f}",
                xref=f"x{i+1} domain" if i > 0 else "x domain",
                yref=f"y{i+1} domain" if i > 0 else "y domain",
                x=0.95, y=0.95,
                xanchor='right', yanchor='top',
                showarrow=False,
                font=dict(size=12, color=color),
                bgcolor='rgba(255,255,255,0.8)',
                row=row, col=col
            )

            # Axis labels
            fig.update_xaxes(title_text=param_name, row=row, col=col)
            if col == 1:
                fig.update_yaxes(title_text=output_name, row=row, col=col)

        fig.update_layout(
            title=dict(
                text=title or f"Parameter Correlations: {output_name}",
                x=0.5,
                font=dict(size=18)
            ),
            height=350 * n_rows,
            width=350 * n_cols,
            showlegend=False
        )

        self._apply_theme(fig)
        return fig

    def create_full_matrix(
        self,
        result: 'MonteCarloResult',
        variables: Optional[List[str]] = None,
        max_points: int = 500
    ) -> go.Figure:
        """
        Create full scatter matrix including inputs and outputs.

        Args:
            result: MonteCarloResult from Monte Carlo run
            variables: Variables to include (default: all inputs + outputs)
            max_points: Maximum points per subplot

        Returns:
            Plotly Figure with NxN scatter matrix
        """
        if variables is None:
            variables = result.parameter_names + result.output_names

        n_vars = len(variables)

        # Collect all data
        data = {}
        for var in variables:
            if var in result.input_samples:
                data[var] = result.input_samples[var]
            elif var in result.output_samples:
                data[var] = result.output_samples[var]
            else:
                raise ValueError(f"Unknown variable: {var}")

        # Subsample
        n_samples = len(list(data.values())[0])
        if n_samples > max_points:
            idx = np.random.choice(n_samples, max_points, replace=False)
            data = {k: v[idx] for k, v in data.items()}

        fig = make_subplots(
            rows=n_vars, cols=n_vars,
            vertical_spacing=0.02,
            horizontal_spacing=0.02
        )

        for i, var_y in enumerate(variables):
            for j, var_x in enumerate(variables):
                row = i + 1
                col = j + 1

                if i == j:
                    # Diagonal: histogram
                    fig.add_trace(
                        go.Histogram(
                            x=data[var_x],
                            nbinsx=20,
                            marker=dict(color=self.colors['histogram']),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                else:
                    # Off-diagonal: scatter
                    corr, _ = stats.spearmanr(data[var_x], data[var_y])
                    color = self.colors['positive_corr'] if corr >= 0 else self.colors['negative_corr']

                    fig.add_trace(
                        go.Scatter(
                            x=data[var_x],
                            y=data[var_y],
                            mode='markers',
                            marker=dict(size=2, color=color, opacity=0.5),
                            showlegend=False
                        ),
                        row=row, col=col
                    )

                # Labels on edges
                if i == n_vars - 1:
                    fig.update_xaxes(title_text=var_x, row=row, col=col, title_font=dict(size=10))
                if j == 0:
                    fig.update_yaxes(title_text=var_y, row=row, col=col, title_font=dict(size=10))

        fig.update_layout(
            title=dict(text="Scatter Matrix", x=0.5),
            height=200 * n_vars,
            width=200 * n_vars,
            showlegend=False
        )

        self._apply_theme(fig)
        return fig


class TornadoPlotter(BaseMCPlotter):
    """
    Creates tornado charts for sensitivity analysis.

    Shows which parameters have the most influence on each output,
    sorted by absolute correlation magnitude.

    Example:
        >>> plotter = TornadoPlotter()
        >>> fig = plotter.create_figure(result, 'isp')
        >>> fig.show()
    """

    def create_figure(
        self,
        result: 'MonteCarloResult',
        output_name: Optional[str] = None,
        correlation_type: str = 'spearman',
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create tornado chart for sensitivity analysis.

        Args:
            result: MonteCarloResult from Monte Carlo run
            output_name: Output to analyze (default: first)
            correlation_type: 'pearson' or 'spearman'
            title: Custom title

        Returns:
            Plotly Figure with horizontal bar chart
        """
        if output_name is None:
            output_name = result.output_names[0]

        if output_name not in result.sensitivity:
            raise ValueError(f"No sensitivity data for: {output_name}")

        sensitivity = result.sensitivity[output_name]
        correlations = sensitivity.get(correlation_type, {})

        if not correlations:
            raise ValueError(f"No {correlation_type} correlations available")

        # Sort by absolute correlation
        sorted_params = sorted(
            correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        param_names = [p[0] for p in sorted_params]
        corr_values = [p[1] for p in sorted_params]

        # Color by sign
        colors = [
            self.colors['positive_corr'] if c >= 0 else self.colors['negative_corr']
            for c in corr_values
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=param_names,
            x=corr_values,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='black', width=1)
            ),
            text=[f"{c:.3f}" for c in corr_values],
            textposition='outside',
            hovertemplate='%{y}: %{x:.4f}<extra></extra>'
        ))

        # Add zero line
        fig.add_vline(x=0, line=dict(color='black', width=1))

        # Layout
        fig.update_layout(
            title=dict(
                text=title or f"Sensitivity Analysis: {output_name}",
                x=0.5,
                font=dict(size=18)
            ),
            xaxis=dict(
                title=f"{correlation_type.capitalize()} Correlation Coefficient",
                range=[-1.1, 1.1],
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=2
            ),
            yaxis=dict(
                title="Parameter",
                autorange="reversed",  # Most important at top
                showgrid=False
            ),
            height=max(400, 50 * len(param_names)),
            width=700,
            showlegend=False,
            margin=dict(l=150)  # Room for parameter names
        )

        # Add annotation for interpretation
        fig.add_annotation(
            text="Positive: parameter increase raises output<br>Negative: parameter increase lowers output",
            xref="paper", yref="paper",
            x=0.5, y=-0.12,
            showarrow=False,
            font=dict(size=10, color='gray'),
            align='center'
        )

        self._apply_theme(fig)
        return fig

    def create_multi_figure(
        self,
        result: 'MonteCarloResult',
        output_names: Optional[List[str]] = None,
        correlation_type: str = 'spearman'
    ) -> go.Figure:
        """
        Create tornado charts for multiple outputs.

        Args:
            result: MonteCarloResult from Monte Carlo run
            output_names: Outputs to analyze (default: all)
            correlation_type: 'pearson' or 'spearman'

        Returns:
            Plotly Figure with subplot grid
        """
        if output_names is None:
            output_names = result.output_names

        n_outputs = len(output_names)

        fig = make_subplots(
            rows=1, cols=n_outputs,
            subplot_titles=output_names,
            horizontal_spacing=0.15
        )

        for i, output_name in enumerate(output_names):
            sensitivity = result.sensitivity.get(output_name, {})
            correlations = sensitivity.get(correlation_type, {})

            sorted_params = sorted(
                correlations.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            param_names = [p[0] for p in sorted_params]
            corr_values = [p[1] for p in sorted_params]

            colors = [
                self.colors['positive_corr'] if c >= 0 else self.colors['negative_corr']
                for c in corr_values
            ]

            fig.add_trace(
                go.Bar(
                    y=param_names,
                    x=corr_values,
                    orientation='h',
                    marker=dict(color=colors),
                    text=[f"{c:.2f}" for c in corr_values],
                    textposition='outside',
                    showlegend=False
                ),
                row=1, col=i+1
            )

            fig.update_xaxes(range=[-1.1, 1.1], row=1, col=i+1)
            fig.update_yaxes(autorange="reversed", row=1, col=i+1)

        fig.update_layout(
            title=dict(
                text="Sensitivity Analysis (Tornado Charts)",
                x=0.5,
                font=dict(size=18)
            ),
            height=max(400, 50 * len(result.parameter_names)),
            width=400 * n_outputs,
            showlegend=False
        )

        self._apply_theme(fig)
        return fig


class ConvergencePlotter(BaseMCPlotter):
    """
    Creates convergence plots showing running statistics vs sample count.

    Helps assess if enough samples have been run for stable results.

    Example:
        >>> plotter = ConvergencePlotter()
        >>> fig = plotter.create_figure(result, 'isp')
        >>> fig.show()
    """

    def create_figure(
        self,
        result: 'MonteCarloResult',
        output_name: Optional[str] = None,
        show_mean: bool = True,
        show_std: bool = True,
        show_percentiles: bool = True,
        n_points: int = 100,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create convergence plot for running statistics.

        Args:
            result: MonteCarloResult from Monte Carlo run
            output_name: Output to analyze (default: first)
            show_mean: Show running mean
            show_std: Show running std (as band around mean)
            show_percentiles: Show running P5/P95
            n_points: Number of points to calculate (for performance)
            title: Custom title

        Returns:
            Plotly Figure with convergence traces
        """
        if output_name is None:
            output_name = result.output_names[0]

        if output_name not in result.output_samples:
            raise ValueError(f"Unknown output: {output_name}")

        samples = result.output_samples[output_name]
        n_samples = len(samples)

        # Calculate at n_points evenly spaced sample sizes
        sample_sizes = np.unique(np.linspace(10, n_samples, n_points).astype(int))

        running_mean = []
        running_std = []
        running_p5 = []
        running_p95 = []

        for n in sample_sizes:
            subset = samples[:n]
            running_mean.append(np.mean(subset))
            running_std.append(np.std(subset))
            running_p5.append(np.percentile(subset, 5))
            running_p95.append(np.percentile(subset, 95))

        running_mean = np.array(running_mean)
        running_std = np.array(running_std)
        running_p5 = np.array(running_p5)
        running_p95 = np.array(running_p95)

        fig = go.Figure()

        # Standard deviation band
        if show_std:
            fig.add_trace(go.Scatter(
                x=np.concatenate([sample_sizes, sample_sizes[::-1]]),
                y=np.concatenate([running_mean + running_std, (running_mean - running_std)[::-1]]),
                fill='toself',
                fillcolor='rgba(52, 152, 219, 0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                name='+/- 1 Std Dev',
                hoverinfo='skip'
            ))

        # Percentile band
        if show_percentiles:
            fig.add_trace(go.Scatter(
                x=np.concatenate([sample_sizes, sample_sizes[::-1]]),
                y=np.concatenate([running_p95, running_p5[::-1]]),
                fill='toself',
                fillcolor='rgba(231, 76, 60, 0.15)',
                line=dict(color='rgba(0,0,0,0)'),
                name='P5-P95 Range',
                hoverinfo='skip'
            ))

            # P5 line
            fig.add_trace(go.Scatter(
                x=sample_sizes,
                y=running_p5,
                mode='lines',
                name='P5',
                line=dict(color=self.colors['p5_p95'], width=1, dash='dash'),
                hovertemplate='Samples: %{x}<br>P5: %{y:.4g}<extra></extra>'
            ))

            # P95 line
            fig.add_trace(go.Scatter(
                x=sample_sizes,
                y=running_p95,
                mode='lines',
                name='P95',
                line=dict(color=self.colors['p5_p95'], width=1, dash='dash'),
                hovertemplate='Samples: %{x}<br>P95: %{y:.4g}<extra></extra>'
            ))

        # Mean line
        if show_mean:
            fig.add_trace(go.Scatter(
                x=sample_sizes,
                y=running_mean,
                mode='lines',
                name='Mean',
                line=dict(color=self.colors['mean'], width=2),
                hovertemplate='Samples: %{x}<br>Mean: %{y:.4g}<extra></extra>'
            ))

        # Final value reference lines
        final_mean = running_mean[-1]
        fig.add_hline(
            y=final_mean,
            line=dict(color='gray', width=1, dash='dot'),
            annotation=dict(
                text=f"Final Mean: {final_mean:.4g}",
                xanchor='left'
            )
        )

        # Layout
        fig.update_layout(
            title=dict(
                text=title or f"Convergence: {output_name}",
                x=0.5,
                font=dict(size=18)
            ),
            xaxis=dict(
                title="Number of Samples",
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(
                title=output_name,
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            showlegend=True,
            legend=dict(
                x=0.98,
                y=0.98,
                xanchor='right',
                bgcolor='rgba(255,255,255,0.9)'
            ),
            height=500,
            width=900
        )

        self._apply_theme(fig)
        return fig

    def create_multi_figure(
        self,
        result: 'MonteCarloResult',
        output_names: Optional[List[str]] = None,
        n_points: int = 50
    ) -> go.Figure:
        """
        Create convergence plots for multiple outputs.

        Args:
            result: MonteCarloResult from Monte Carlo run
            output_names: Outputs to plot (default: all)
            n_points: Number of convergence points

        Returns:
            Plotly Figure with subplot grid
        """
        if output_names is None:
            output_names = result.output_names

        n_outputs = len(output_names)
        n_cols = min(2, n_outputs)
        n_rows = int(np.ceil(n_outputs / n_cols))

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=output_names,
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        for i, output_name in enumerate(output_names):
            row = i // n_cols + 1
            col = i % n_cols + 1

            samples = result.output_samples[output_name]
            n_samples = len(samples)
            sample_sizes = np.unique(np.linspace(10, n_samples, n_points).astype(int))

            running_mean = [np.mean(samples[:n]) for n in sample_sizes]
            running_std = [np.std(samples[:n]) for n in sample_sizes]

            running_mean = np.array(running_mean)
            running_std = np.array(running_std)

            # Std band
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([sample_sizes, sample_sizes[::-1]]),
                    y=np.concatenate([running_mean + running_std, (running_mean - running_std)[::-1]]),
                    fill='toself',
                    fillcolor='rgba(52, 152, 219, 0.2)',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

            # Mean line
            fig.add_trace(
                go.Scatter(
                    x=sample_sizes,
                    y=running_mean,
                    mode='lines',
                    line=dict(color=self.colors['mean'], width=2),
                    showlegend=False,
                    hovertemplate=f'{output_name}<br>Samples: %{{x}}<br>Mean: %{{y:.4g}}<extra></extra>'
                ),
                row=row, col=col
            )

            fig.update_xaxes(title_text="Samples", row=row, col=col)
            fig.update_yaxes(title_text=output_name, row=row, col=col)

        fig.update_layout(
            title=dict(
                text="Convergence Analysis",
                x=0.5,
                font=dict(size=18)
            ),
            height=350 * n_rows,
            width=500 * n_cols,
            showlegend=False
        )

        self._apply_theme(fig)
        return fig
