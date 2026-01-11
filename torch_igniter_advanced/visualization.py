"""
Visualization tools for torch igniter analysis.

Creates interactive Plotly charts for performance data and operating envelopes.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict


def plot_mixture_ratio_sweep(
    df: pd.DataFrame,
    title: str = "Performance vs. Mixture Ratio"
) -> go.Figure:
    """Plot performance metrics vs. mixture ratio.
    
    Args:
        df: DataFrame from EnvelopeGenerator.generate_mixture_ratio_sweep()
        title: Plot title
        
    Returns:
        Plotly figure with subplots
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'C* and Isp',
            'Flame Temperature',
            'Thrust and Heat Power',
            'Mass Flows'
        ),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # Row 1, Col 1: C* and Isp
    fig.add_trace(
        go.Scatter(x=df['mixture_ratio'], y=df['c_star'],
                   name='C*', line=dict(color='blue')),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df['mixture_ratio'], y=df['isp'],
                   name='Isp', line=dict(color='red')),
        row=1, col=1, secondary_y=True
    )
    
    # Row 1, Col 2: Flame Temperature
    fig.add_trace(
        go.Scatter(x=df['mixture_ratio'], y=df['flame_temperature'],
                   name='T_flame', line=dict(color='orange'),
                   showlegend=False),
        row=1, col=2
    )
    
    # Row 2, Col 1: Thrust and Heat Power
    fig.add_trace(
        go.Scatter(x=df['mixture_ratio'], y=df['thrust'],
                   name='Thrust', line=dict(color='green')),
        row=2, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df['mixture_ratio'], y=df['heat_power'],
                   name='Heat Power', line=dict(color='purple')),
        row=2, col=1, secondary_y=True
    )
    
    # Row 2, Col 2: Mass Flows
    fig.add_trace(
        go.Scatter(x=df['mixture_ratio'], y=df['n2o_mass_flow'],
                   name='N2O', line=dict(color='cyan'),
                   fill='tonexty'),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['mixture_ratio'], y=df['ethanol_mass_flow'],
                   name='Ethanol', line=dict(color='brown'),
                   fill='tozeroy'),
        row=2, col=2
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Mixture Ratio (O/F)", row=1, col=1)
    fig.update_xaxes(title_text="Mixture Ratio (O/F)", row=1, col=2)
    fig.update_xaxes(title_text="Mixture Ratio (O/F)", row=2, col=1)
    fig.update_xaxes(title_text="Mixture Ratio (O/F)", row=2, col=2)
    
    fig.update_yaxes(title_text="C* (m/s)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Isp (s)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Temperature (K)", row=1, col=2)
    fig.update_yaxes(title_text="Thrust (N)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Heat Power (kW)", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Mass Flow (g/s)", row=2, col=2)
    
    fig.update_layout(
        title_text=title,
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def plot_pressure_sweep(
    df: pd.DataFrame,
    title: str = "Performance vs. Chamber Pressure"
) -> go.Figure:
    """Plot performance metrics vs. chamber pressure.
    
    Args:
        df: DataFrame from EnvelopeGenerator.generate_pressure_sweep()
        title: Plot title
        
    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'C* and Isp',
            'Flame Temperature and Gamma',
            'Thrust and Heat Power',
            'Specific Impulse vs Pressure'
        ),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # Row 1, Col 1: C* and Isp
    fig.add_trace(
        go.Scatter(x=df['chamber_pressure'], y=df['c_star'],
                   name='C*', line=dict(color='blue')),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df['chamber_pressure'], y=df['isp'],
                   name='Isp', line=dict(color='red')),
        row=1, col=1, secondary_y=True
    )
    
    # Row 1, Col 2: Temperature and Gamma
    fig.add_trace(
        go.Scatter(x=df['chamber_pressure'], y=df['flame_temperature'],
                   name='T_flame', line=dict(color='orange')),
        row=1, col=2, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df['chamber_pressure'], y=df['gamma'],
                   name='Gamma', line=dict(color='green')),
        row=1, col=2, secondary_y=True
    )
    
    # Row 2, Col 1: Thrust and Heat Power
    fig.add_trace(
        go.Scatter(x=df['chamber_pressure'], y=df['thrust'],
                   name='Thrust', line=dict(color='green')),
        row=2, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df['chamber_pressure'], y=df['heat_power'],
                   name='Heat Power', line=dict(color='purple')),
        row=2, col=1, secondary_y=True
    )
    
    # Row 2, Col 2: Isp detailed
    fig.add_trace(
        go.Scatter(x=df['chamber_pressure'], y=df['isp'],
                   name='Isp', line=dict(color='red', width=3),
                   showlegend=False),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Chamber Pressure (bar)", row=1, col=1)
    fig.update_xaxes(title_text="Chamber Pressure (bar)", row=1, col=2)
    fig.update_xaxes(title_text="Chamber Pressure (bar)", row=2, col=1)
    fig.update_xaxes(title_text="Chamber Pressure (bar)", row=2, col=2)
    
    fig.update_yaxes(title_text="C* (m/s)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Isp (s)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Temperature (K)", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Gamma", row=1, col=2, secondary_y=True)
    fig.update_yaxes(title_text="Thrust (N)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Heat Power (kW)", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Isp (s)", row=2, col=2)
    
    fig.update_layout(
        title_text=title,
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def plot_2d_envelope(
    df: pd.DataFrame,
    parameter: str = 'isp',
    title: Optional[str] = None
) -> go.Figure:
    """Create 2D contour plot of operating envelope.
    
    Args:
        df: DataFrame from EnvelopeGenerator.generate_2d_envelope()
        parameter: Column to plot ('isp', 'c_star', 'flame_temperature', etc.)
        title: Plot title (auto-generated if None)
        
    Returns:
        Plotly figure with contour plot
    """
    if parameter not in df.columns:
        raise ValueError(f"Parameter '{parameter}' not in DataFrame")
    
    # Pivot for contour plot
    pivot = df.pivot_table(
        values=parameter,
        index='chamber_pressure',
        columns='mixture_ratio',
        aggfunc='mean'
    )
    
    # Create contour plot
    fig = go.Figure(data=go.Contour(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Viridis',
        contours=dict(
            showlabels=True,
            labelfont=dict(size=10, color='white')
        ),
        colorbar=dict(title=parameter)
    ))
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df['mixture_ratio'],
        y=df['chamber_pressure'],
        mode='markers',
        marker=dict(size=3, color='white', opacity=0.3),
        showlegend=False,
        hovertemplate='MR: %{x:.2f}<br>P: %{y:.1f} bar<extra></extra>'
    ))
    
    if title is None:
        title = f"{parameter.replace('_', ' ').title()} Operating Envelope"
    
    fig.update_layout(
        title=title,
        xaxis_title="Mixture Ratio (O/F)",
        yaxis_title="Chamber Pressure (bar)",
        height=600,
        hovermode='closest'
    )
    
    return fig


def plot_geometry_schematic(
    results: 'IgniterResults'
) -> go.Figure:
    """Create schematic diagram of igniter geometry.
    
    Args:
        results: IgniterResults object
        
    Returns:
        Plotly figure with geometry schematic
    """
    # Convert to mm
    chamber_dia = results.chamber_diameter * 1000
    chamber_len = results.chamber_length * 1000
    throat_dia = results.throat_diameter * 1000
    exit_dia = results.exit_diameter * 1000
    nozzle_len = results.nozzle_length * 1000
    
    # Create chamber outline
    chamber_x = [0, 0, chamber_len, chamber_len, 0]
    chamber_y = [chamber_dia/2, -chamber_dia/2, -chamber_dia/2, chamber_dia/2, chamber_dia/2]
    
    # Create nozzle outline (simplified conical)
    nozzle_x = [
        chamber_len,
        chamber_len + nozzle_len/3,
        chamber_len + nozzle_len,
        chamber_len + nozzle_len,
        chamber_len + nozzle_len/3,
        chamber_len
    ]
    nozzle_y = [
        chamber_dia/2,
        throat_dia/2,
        exit_dia/2,
        -exit_dia/2,
        -throat_dia/2,
        -chamber_dia/2
    ]
    
    fig = go.Figure()
    
    # Chamber
    fig.add_trace(go.Scatter(
        x=chamber_x, y=chamber_y,
        fill='toself',
        fillcolor='lightblue',
        line=dict(color='blue', width=2),
        name='Chamber',
        hovertemplate='Chamber<extra></extra>'
    ))
    
    # Nozzle
    fig.add_trace(go.Scatter(
        x=nozzle_x, y=nozzle_y,
        fill='toself',
        fillcolor='lightcoral',
        line=dict(color='red', width=2),
        name='Nozzle',
        hovertemplate='Nozzle<extra></extra>'
    ))
    
    # Centerline
    fig.add_trace(go.Scatter(
        x=[0, chamber_len + nozzle_len],
        y=[0, 0],
        line=dict(color='black', dash='dash'),
        name='Centerline',
        showlegend=False
    ))
    
    # Annotations
    annotations = [
        dict(x=chamber_len/2, y=chamber_dia/2 + 5,
             text=f"L = {chamber_len:.1f} mm",
             showarrow=False, font=dict(size=10)),
        dict(x=-5, y=0,
             text=f"D = {chamber_dia:.1f} mm",
             showarrow=False, font=dict(size=10), textangle=-90),
        dict(x=chamber_len + nozzle_len/3, y=0,
             text=f"D* = {throat_dia:.1f} mm",
             showarrow=True, arrowhead=2, ax=0, ay=-30, font=dict(size=10)),
        dict(x=chamber_len + nozzle_len, y=exit_dia/2 + 5,
             text=f"D_exit = {exit_dia:.1f} mm",
             showarrow=False, font=dict(size=10)),
    ]
    
    fig.update_layout(
        title="Igniter Geometry",
        xaxis=dict(title="Axial Position (mm)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Radial Position (mm)"),
        height=400,
        showlegend=True,
        annotations=annotations,
        hovermode='closest'
    )
    
    return fig


def plot_mass_flow_breakdown(
    results: 'IgniterResults'
) -> go.Figure:
    """Create pie chart of mass flow breakdown.
    
    Args:
        results: IgniterResults object
        
    Returns:
        Plotly figure with pie chart
    """
    labels = ['N2O (Oxidizer)', 'Ethanol (Fuel)']
    values = [
        results.oxidizer_mass_flow * 1000,  # g/s
        results.fuel_mass_flow * 1000
    ]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker=dict(colors=['cyan', 'brown']),
        textinfo='label+value+percent',
        hovertemplate='%{label}<br>%{value:.2f} g/s<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=f"Mass Flow Breakdown (Total: {sum(values):.1f} g/s, MR: {results.mixture_ratio:.2f})",
        height=400
    )
    
    return fig


def plot_performance_summary(
    results: 'IgniterResults'
) -> go.Figure:
    """Create summary bar chart of key performance metrics.
    
    Args:
        results: IgniterResults object
        
    Returns:
        Plotly figure with bar charts
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Combustion', 'Performance', 'Injector Sizing')
    )
    
    # Combustion metrics (normalized to show on same scale)
    combustion = {
        'T_flame': results.flame_temperature,
        'C*': results.c_star,
        'Gamma': results.gamma * 1000  # Scale up for visibility
    }
    
    fig.add_trace(
        go.Bar(x=list(combustion.keys()), y=list(combustion.values()),
               marker_color=['orange', 'blue', 'green'],
               showlegend=False,
               text=[f"{v:.0f}" for v in combustion.values()],
               textposition='outside'),
        row=1, col=1
    )
    
    # Performance metrics
    performance = {
        'Isp (s)': results.isp_theoretical,
        'Thrust (N)': results.thrust,
        'Power (kW)': results.heat_power_output / 1000
    }
    
    fig.add_trace(
        go.Bar(x=list(performance.keys()), y=list(performance.values()),
               marker_color=['red', 'green', 'purple'],
               showlegend=False,
               text=[f"{v:.1f}" for v in performance.values()],
               textposition='outside'),
        row=1, col=2
    )
    
    # Injector sizing
    injector = {
        'N2O (mm)': results.n2o_orifice_diameter * 1000,
        'EtOH (mm)': results.ethanol_orifice_diameter * 1000
    }
    
    fig.add_trace(
        go.Bar(x=list(injector.keys()), y=list(injector.values()),
               marker_color=['cyan', 'brown'],
               showlegend=False,
               text=[f"{v:.2f}" for v in injector.values()],
               textposition='outside'),
        row=1, col=3
    )
    
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=2)
    fig.update_yaxes(title_text="Diameter (mm)", row=1, col=3)
    
    fig.update_layout(
        title_text="Performance Summary",
        height=400,
        showlegend=False
    )
    
    return fig


def save_figure(fig: go.Figure, filepath: str, format: str = 'html'):
    """Save Plotly figure to file.
    
    Args:
        fig: Plotly figure
        filepath: Output filepath
        format: 'html', 'png', 'svg', or 'pdf'
    """
    if format == 'html':
        fig.write_html(filepath)
    elif format in ['png', 'svg', 'pdf']:
        fig.write_image(filepath)
    else:
        raise ValueError(f"Unknown format: {format}")
