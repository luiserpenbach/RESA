"""
Visualization utilities for swirl injector analysis.

Uses Plotly for interactive plots.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, List
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

from results import InjectorResults, ColdFlowResults
from config import InjectorConfig
from thermodynamics import DischargeCoefficients, SprayAngleCorrelations


def _check_plotly():
    """Check if plotly is available and raise helpful error if not."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for visualization. "
            "Install with: pip install plotly"
        )


def plot_injector_cross_section(
    results: InjectorResults,
    title: Optional[str] = None,
    show: bool = True
) -> "go.Figure":
    """
    Plot schematic cross-section of injector element.
    
    Args:
        results: Injector calculation results
        title: Optional plot title
        show: Whether to display the figure
        
    Returns:
        Plotly figure object
    """
    _check_plotly()
    
    geom = results.geometry
    
    # Convert to mm for display
    r_o = geom.orifice_radius * 1000
    r_sc = geom.swirl_chamber_radius * 1000
    r_ox = geom.oxidizer_port_radius * 1000
    r_p = geom.port_radius * 1000
    
    # Lengths (estimate if not provided)
    l_o = (geom.orifice_length or geom.orifice_radius) * 1000
    l_sc = (geom.swirl_chamber_length or 2 * geom.swirl_chamber_radius) * 1000
    
    fig = go.Figure()
    
    # Draw swirl chamber
    fig.add_shape(type="rect",
        x0=0, y0=-r_sc, x1=l_sc, y1=r_sc,
        line=dict(color="blue", width=2),
        fillcolor="lightblue", opacity=0.3
    )
    
    # Draw orifice
    fig.add_shape(type="rect",
        x0=l_sc, y0=-r_o, x1=l_sc + l_o, y1=r_o,
        line=dict(color="blue", width=2),
        fillcolor="lightblue", opacity=0.5
    )
    
    # Draw oxidizer port (simplified)
    fig.add_shape(type="rect",
        x0=l_sc + l_o, y0=-r_ox, x1=l_sc + l_o + 5, y1=r_ox,
        line=dict(color="red", width=2),
        fillcolor="lightyellow", opacity=0.3
    )
    
    # Draw tangential port
    fig.add_shape(type="circle",
        x0=-r_p, y0=r_sc - 2*r_p, x1=r_p, y1=r_sc,
        line=dict(color="green", width=2),
        fillcolor="lightgreen", opacity=0.5
    )
    
    # Add annotations
    fig.add_annotation(x=l_sc/2, y=r_sc + 1, text=f"Swirl Chamber: Ø{2*r_sc:.2f} mm",
                      showarrow=False, font=dict(size=10))
    fig.add_annotation(x=l_sc + l_o/2, y=r_o + 1, text=f"Orifice: Ø{2*r_o:.2f} mm",
                      showarrow=False, font=dict(size=10))
    fig.add_annotation(x=l_sc + l_o + 2.5, y=r_ox + 1, text=f"Ox Port: Ø{2*r_ox:.2f} mm",
                      showarrow=False, font=dict(size=10))
    
    fig.update_layout(
        title=title or f"{results.injector_type} Injector Cross-Section",
        xaxis_title="Axial Position (mm)",
        yaxis_title="Radial Position (mm)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=False,
        height=400
    )
    
    if show:
        fig.show()
    
    return fig


def plot_discharge_coefficient_comparison(
    swirl_numbers: np.ndarray = None,
    show: bool = True
) -> "go.Figure":
    """
    Compare discharge coefficient correlations across swirl numbers.
    
    Args:
        swirl_numbers: Array of swirl numbers to evaluate
        show: Whether to display the figure
        
    Returns:
        Plotly figure object
    """
    _check_plotly()
    if swirl_numbers is None:
        swirl_numbers = np.linspace(1, 20, 100)
    
    # For each swirl number, back-calculate geometry assuming r_sc = 1, n_p = 3
    r_sc = 1.0
    n_p = 3
    
    cd_abramovic = []
    cd_fu = []
    cd_anand = []
    
    for SN in swirl_numbers:
        # From SN = (r_sc - r_p) * r_sc / (n_p * r_p^2)
        # Solve quadratic for r_p
        # n_p * SN * r_p^2 + r_sc * r_p - r_sc^2 = 0
        a = n_p * SN
        b = r_sc
        c = -r_sc ** 2
        r_p = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        
        cd_abramovic.append(DischargeCoefficients.abramovic(r_sc, r_p, n_p))
        cd_fu.append(DischargeCoefficients.fu(r_sc, r_p, n_p))
        cd_anand.append(DischargeCoefficients.anand(r_sc, r_p, n_p))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=swirl_numbers, y=cd_abramovic,
        mode='lines', name='Abramovic',
        line=dict(width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=swirl_numbers, y=cd_fu,
        mode='lines', name='Fu et al.',
        line=dict(width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=swirl_numbers, y=cd_anand,
        mode='lines', name='Anand et al.',
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title="Discharge Coefficient Correlations Comparison",
        xaxis_title="Swirl Number",
        yaxis_title="Discharge Coefficient",
        legend=dict(x=0.7, y=0.95),
        height=500
    )
    
    if show:
        fig.show()
    
    return fig


def plot_spray_angle_comparison(
    swirl_numbers: np.ndarray = None,
    show: bool = True
) -> "go.Figure":
    """
    Compare spray angle correlations across swirl numbers.
    
    Args:
        swirl_numbers: Array of swirl numbers to evaluate
        show: Whether to display the figure
        
    Returns:
        Plotly figure object
    """
    if swirl_numbers is None:
        swirl_numbers = np.linspace(1, 20, 100)
    
    r_sc = 1.0
    n_p = 3
    
    alpha_anand = []
    
    for SN in swirl_numbers:
        a = n_p * SN
        b = r_sc
        c = -r_sc ** 2
        r_p = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        
        alpha = SprayAngleCorrelations.anand(r_sc, r_p, n_p, 1.5)
        alpha_anand.append(np.rad2deg(alpha))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=swirl_numbers, y=alpha_anand,
        mode='lines', name='Anand et al. (RR=1.5)',
        line=dict(width=2, color='blue')
    ))
    
    # Add Lefebvre correlation for comparison
    X_range = np.linspace(0.1, 0.9, 100)
    alpha_lefebvre = [np.rad2deg(SprayAngleCorrelations.lefebvre(X)) for X in X_range]
    
    fig.add_trace(go.Scatter(
        x=X_range * 20,  # Scale to similar range
        y=alpha_lefebvre,
        mode='lines', name='Lefebvre (vs X scaled)',
        line=dict(width=2, color='red', dash='dash')
    ))
    
    fig.update_layout(
        title="Spray Angle Correlations Comparison",
        xaxis_title="Swirl Number",
        yaxis_title="Spray Half Angle (°)",
        legend=dict(x=0.6, y=0.95),
        height=500
    )
    
    if show:
        fig.show()
    
    return fig


def plot_sensitivity_analysis(
    config: InjectorConfig,
    parameter: str,
    values: np.ndarray,
    calculator_class,
    metrics: List[str] = None,
    show: bool = True
) -> "go.Figure":
    """
    Plot sensitivity of output metrics to a design parameter.
    
    Args:
        config: Base injector configuration
        parameter: Parameter to vary (dot notation for nested, e.g., 'operating.pressure_drop')
        values: Array of values to test
        calculator_class: Calculator class to use (LCSCCalculator or GCSCCalculator)
        metrics: List of metrics to plot (default: J, We, spray_half_angle)
        show: Whether to display the figure
        
    Returns:
        Plotly figure object
    """
    if metrics is None:
        metrics = ['momentum_flux_ratio', 'weber_number', 'spray_half_angle']
    
    results_data = {metric: [] for metric in metrics}
    
    for val in values:
        # Create modified config
        config_dict = config.to_dict()
        
        # Navigate to nested parameter
        parts = parameter.split('.')
        target = config_dict
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = val
        
        # Recalculate
        modified_config = InjectorConfig.from_dict(config_dict)
        calc = calculator_class(modified_config)
        result = calc.calculate()
        
        for metric in metrics:
            val_metric = getattr(result.performance, metric)
            results_data[metric].append(val_metric)
    
    # Create subplots
    fig = make_subplots(
        rows=len(metrics), cols=1,
        subplot_titles=[m.replace('_', ' ').title() for m in metrics],
        vertical_spacing=0.1
    )
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Scatter(
                x=values, y=results_data[metric],
                mode='lines+markers',
                name=metric.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ),
            row=i+1, col=1
        )
    
    param_label = parameter.split('.')[-1].replace('_', ' ').title()
    fig.update_layout(
        title=f"Sensitivity Analysis: {param_label}",
        height=300 * len(metrics),
        showlegend=False
    )
    
    # Update x-axis labels
    for i in range(len(metrics)):
        fig.update_xaxes(title_text=param_label, row=i+1, col=1)
    
    if show:
        fig.show()
    
    return fig


def plot_operating_envelope(
    config: InjectorConfig,
    calculator_class,
    pressure_drops: np.ndarray = None,
    mass_flows: np.ndarray = None,
    show: bool = True
) -> "go.Figure":
    """
    Plot operating envelope showing valid design space.
    
    Args:
        config: Base injector configuration
        calculator_class: Calculator class to use
        pressure_drops: Array of pressure drops to evaluate
        mass_flows: Array of fuel mass flows to evaluate
        show: Whether to display the figure
        
    Returns:
        Plotly figure object
    """
    if pressure_drops is None:
        pressure_drops = np.linspace(5e5, 30e5, 20)
    if mass_flows is None:
        mass_flows = np.linspace(0.1, 0.5, 20)
    
    J_map = np.zeros((len(pressure_drops), len(mass_flows)))
    We_map = np.zeros((len(pressure_drops), len(mass_flows)))
    
    for i, dp in enumerate(pressure_drops):
        for j, mf in enumerate(mass_flows):
            try:
                config_dict = config.to_dict()
                config_dict['operating']['pressure_drop'] = dp
                config_dict['operating']['mass_flow_fuel'] = mf
                modified_config = InjectorConfig.from_dict(config_dict)
                
                calc = calculator_class(modified_config)
                result = calc.calculate()
                
                J_map[i, j] = result.performance.momentum_flux_ratio
                We_map[i, j] = result.performance.weber_number
            except:
                J_map[i, j] = np.nan
                We_map[i, j] = np.nan
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Momentum Flux Ratio (J)', 'Weber Number'],
        horizontal_spacing=0.15
    )
    
    fig.add_trace(
        go.Heatmap(
            x=mass_flows,
            y=pressure_drops / 1e5,
            z=J_map,
            colorscale='Viridis',
            colorbar=dict(title='J', x=0.45)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Heatmap(
            x=mass_flows,
            y=pressure_drops / 1e5,
            z=We_map,
            colorscale='Plasma',
            colorbar=dict(title='We', x=1.0)
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Fuel Mass Flow (kg/s)", row=1, col=1)
    fig.update_xaxes(title_text="Fuel Mass Flow (kg/s)", row=1, col=2)
    fig.update_yaxes(title_text="Pressure Drop (bar)", row=1, col=1)
    fig.update_yaxes(title_text="Pressure Drop (bar)", row=1, col=2)
    
    fig.update_layout(
        title="Operating Envelope",
        height=500
    )
    
    if show:
        fig.show()
    
    return fig


def save_figure(fig: "go.Figure", path: str | Path, format: str = 'html') -> None:
    """
    Save Plotly figure to file.
    
    Args:
        fig: Plotly figure object
        path: Output file path
        format: Output format ('html', 'png', 'svg', 'pdf')
    """
    _check_plotly()
    path = Path(path)
    
    if format == 'html':
        fig.write_html(str(path))
    else:
        fig.write_image(str(path), format=format)
