"""
HTML report generation for torch igniter analysis.

Creates comprehensive HTML reports with embedded interactive charts.
"""

from typing import Optional, List
from datetime import datetime
from pathlib import Path
import base64

from .config import IgniterConfig, IgniterResults
from .analysis import EnvelopeGenerator
from .visualization import (
    plot_mixture_ratio_sweep,
    plot_pressure_sweep,
    plot_geometry_schematic,
    plot_mass_flow_breakdown,
    plot_performance_summary
)


class ReportGenerator:
    """Generate HTML reports for igniter analysis."""
    
    def __init__(self):
        """Initialize report generator."""
        self.envelope_gen = EnvelopeGenerator()
    
    def generate_design_report(
        self,
        config: IgniterConfig,
        results: IgniterResults,
        output_path: str,
        include_envelopes: bool = True
    ) -> str:
        """Generate comprehensive design report.
        
        Args:
            config: Input configuration
            results: Design results
            output_path: Output HTML filepath
            include_envelopes: Whether to include operating envelopes
            
        Returns:
            Path to generated HTML file
        """
        html_parts = []
        
        # Header
        html_parts.append(self._generate_header(config.name))
        
        # Summary section
        html_parts.append(self._generate_summary_section(config, results))
        
        # Geometry visualization
        html_parts.append(self._generate_geometry_section(results))
        
        # Performance metrics
        html_parts.append(self._generate_performance_section(results))
        
        # Detailed design data
        html_parts.append(self._generate_detailed_section(config, results))
        
        # Operating envelopes (if requested)
        if include_envelopes:
            html_parts.append(self._generate_envelope_section(config))
        
        # Footer
        html_parts.append(self._generate_footer())
        
        # Combine and write
        html_content = '\n'.join(html_parts)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_header(self, title: str) -> str:
        """Generate HTML header with CSS."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Design Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .timestamp {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        .metric-unit {{
            font-size: 0.8em;
            color: #666;
            margin-left: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .chart-container {{
            margin: 20px 0;
            background: white;
            padding: 15px;
            border-radius: 8px;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .info {{
            background-color: #d1ecf1;
            border-left: 4px solid #0c5460;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
    
    def _generate_summary_section(self, config: IgniterConfig, results: IgniterResults) -> str:
        """Generate summary section with key metrics."""
        return f"""
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Chamber Pressure</div>
                <div class="metric-value">{results.chamber_pressure/1e5:.1f}<span class="metric-unit">bar</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Mixture Ratio (O/F)</div>
                <div class="metric-value">{results.mixture_ratio:.2f}<span class="metric-unit"></span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Mass Flow</div>
                <div class="metric-value">{results.total_mass_flow*1000:.1f}<span class="metric-unit">g/s</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Heat Power Output</div>
                <div class="metric-value">{results.heat_power_output/1000:.1f}<span class="metric-unit">kW</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Specific Impulse</div>
                <div class="metric-value">{results.isp_theoretical:.1f}<span class="metric-unit">s</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Thrust</div>
                <div class="metric-value">{results.thrust:.1f}<span class="metric-unit">N</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Flame Temperature</div>
                <div class="metric-value">{results.flame_temperature:.0f}<span class="metric-unit">K</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">C* Efficiency</div>
                <div class="metric-value">{results.c_star_efficiency*100:.1f}<span class="metric-unit">%</span></div>
            </div>
        </div>
        
        <div class="info">
            <strong>Configuration:</strong> {config.description if config.description else 'Standard torch igniter design'}
        </div>
    </div>
"""
    
    def _generate_geometry_section(self, results: IgniterResults) -> str:
        """Generate geometry section with schematic."""
        fig_geometry = plot_geometry_schematic(results)
        fig_mass_flow = plot_mass_flow_breakdown(results)
        
        return f"""
    <div class="section">
        <h2>Geometry & Configuration</h2>
        <div class="chart-container">
            {fig_geometry.to_html(include_plotlyjs=False, div_id="geometry")}
        </div>
        
        <table>
            <tr>
                <th>Component</th>
                <th>Diameter (mm)</th>
                <th>Length (mm)</th>
                <th>Area (mm²)</th>
            </tr>
            <tr>
                <td>Chamber</td>
                <td>{results.chamber_diameter*1000:.2f}</td>
                <td>{results.chamber_length*1000:.2f}</td>
                <td>{results.chamber_volume*1e9:.2f} (volume: mm³)</td>
            </tr>
            <tr>
                <td>Throat</td>
                <td>{results.throat_diameter*1000:.2f}</td>
                <td>-</td>
                <td>{results.throat_area*1e6:.2f}</td>
            </tr>
            <tr>
                <td>Exit</td>
                <td>{results.exit_diameter*1000:.2f}</td>
                <td>-</td>
                <td>{results.exit_area*1e6:.2f}</td>
            </tr>
            <tr>
                <td>Nozzle</td>
                <td>-</td>
                <td>{results.nozzle_length*1000:.2f}</td>
                <td>-</td>
            </tr>
        </table>
        
        <div class="chart-container">
            {fig_mass_flow.to_html(include_plotlyjs=False, div_id="mass_flow")}
        </div>
    </div>
"""
    
    def _generate_performance_section(self, results: IgniterResults) -> str:
        """Generate performance section."""
        fig_summary = plot_performance_summary(results)
        
        return f"""
    <div class="section">
        <h2>Performance Metrics</h2>
        <div class="chart-container">
            {fig_summary.to_html(include_plotlyjs=False, div_id="performance")}
        </div>
        
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
                <th>Units</th>
            </tr>
            <tr>
                <td>Characteristic Velocity (C*)</td>
                <td>{results.c_star:.1f}</td>
                <td>m/s</td>
            </tr>
            <tr>
                <td>Specific Impulse (Isp)</td>
                <td>{results.isp_theoretical:.1f}</td>
                <td>s</td>
            </tr>
            <tr>
                <td>Flame Temperature</td>
                <td>{results.flame_temperature:.0f}</td>
                <td>K ({results.flame_temperature-273.15:.0f} °C)</td>
            </tr>
            <tr>
                <td>Gamma (specific heat ratio)</td>
                <td>{results.gamma:.3f}</td>
                <td>-</td>
            </tr>
            <tr>
                <td>Molecular Weight</td>
                <td>{results.molecular_weight:.2f}</td>
                <td>kg/kmol</td>
            </tr>
            <tr>
                <td>Thrust</td>
                <td>{results.thrust:.2f}</td>
                <td>N</td>
            </tr>
            <tr>
                <td>Heat Power Output</td>
                <td>{results.heat_power_output/1000:.2f}</td>
                <td>kW</td>
            </tr>
        </table>
    </div>
"""
    
    def _generate_detailed_section(self, config: IgniterConfig, results: IgniterResults) -> str:
        """Generate detailed design data section."""
        return f"""
    <div class="section">
        <h2>Injector Design</h2>
        <table>
            <tr>
                <th>Propellant</th>
                <th>Orifice Diameter (mm)</th>
                <th>Injection Velocity (m/s)</th>
                <th>Pressure Drop (bar)</th>
                <th>Mass Flow (g/s)</th>
            </tr>
            <tr>
                <td>N2O (Oxidizer)</td>
                <td>{results.n2o_orifice_diameter*1000:.3f}</td>
                <td>{results.n2o_injection_velocity:.1f}</td>
                <td>{results.n2o_pressure_drop/1e5:.2f}</td>
                <td>{results.oxidizer_mass_flow*1000:.2f}</td>
            </tr>
            <tr>
                <td>Ethanol (Fuel)</td>
                <td>{results.ethanol_orifice_diameter*1000:.3f}</td>
                <td>{results.ethanol_injection_velocity:.1f}</td>
                <td>{results.ethanol_pressure_drop/1e5:.2f}</td>
                <td>{results.fuel_mass_flow*1000:.2f}</td>
            </tr>
        </table>
        
        <div class="info">
            <strong>Note:</strong> N2O sizing uses Homogeneous Equilibrium Model (HEM) for two-phase flow.
            Ethanol uses incompressible flow model.
        </div>
    </div>
    
    <div class="section">
        <h2>Design Parameters</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>L* (characteristic length)</td>
                <td>{config.l_star:.2f} m</td>
            </tr>
            <tr>
                <td>Expansion Ratio</td>
                <td>{config.expansion_ratio:.1f}</td>
            </tr>
            <tr>
                <td>Nozzle Type</td>
                <td>{config.nozzle_type.title()}</td>
            </tr>
            <tr>
                <td>N2O Orifice Count</td>
                <td>{config.n2o_orifice_count}</td>
            </tr>
            <tr>
                <td>Ethanol Orifice Count</td>
                <td>{config.ethanol_orifice_count}</td>
            </tr>
            <tr>
                <td>Discharge Coefficient</td>
                <td>{config.discharge_coefficient:.2f}</td>
            </tr>
            <tr>
                <td>N2O Feed Pressure</td>
                <td>{config.n2o_feed_pressure/1e5:.1f} bar</td>
            </tr>
            <tr>
                <td>Ethanol Feed Pressure</td>
                <td>{config.ethanol_feed_pressure/1e5:.1f} bar</td>
            </tr>
            <tr>
                <td>N2O Feed Temperature</td>
                <td>{config.n2o_feed_temperature:.1f} K ({config.n2o_feed_temperature-273.15:.1f} °C)</td>
            </tr>
            <tr>
                <td>Ethanol Feed Temperature</td>
                <td>{config.ethanol_feed_temperature:.1f} K ({config.ethanol_feed_temperature-273.15:.1f} °C)</td>
            </tr>
        </table>
    </div>
"""
    
    def _generate_envelope_section(self, config: IgniterConfig) -> str:
        """Generate operating envelope section."""
        # Generate sweeps
        mr_sweep = self.envelope_gen.generate_mixture_ratio_sweep(
            config, mr_range=(1.5, 3.0), n_points=20
        )
        
        p_sweep = self.envelope_gen.generate_pressure_sweep(
            config, pressure_range=(10e5, 30e5), n_points=20
        )
        
        # Create plots
        fig_mr = plot_mixture_ratio_sweep(mr_sweep)
        fig_p = plot_pressure_sweep(p_sweep)
        
        return f"""
    <div class="section">
        <h2>Operating Envelopes</h2>
        <p>The following charts show performance variation across different operating conditions.</p>
        
        <h3>Mixture Ratio Sweep</h3>
        <div class="chart-container">
            {fig_mr.to_html(include_plotlyjs=False, div_id="mr_sweep")}
        </div>
        
        <h3>Pressure Sweep</h3>
        <div class="chart-container">
            {fig_p.to_html(include_plotlyjs=False, div_id="p_sweep")}
        </div>
    </div>
"""
    
    def _generate_footer(self) -> str:
        """Generate HTML footer."""
        return """
    <div class="section" style="text-align: center; color: #666;">
        <p>Generated by Torch Igniter Sizing Tool</p>
        <p style="font-size: 0.9em;">© 2026 - Ethanol/N2O Bipropellant Igniter Design</p>
    </div>
</body>
</html>
"""


def generate_quick_report(
    config: IgniterConfig,
    results: IgniterResults,
    output_path: str
) -> str:
    """Generate quick design report without operating envelopes.
    
    Args:
        config: Input configuration
        results: Design results
        output_path: Output HTML filepath
        
    Returns:
        Path to generated HTML file
    """
    generator = ReportGenerator()
    return generator.generate_design_report(
        config, results, output_path, include_envelopes=False
    )


def generate_full_report(
    config: IgniterConfig,
    results: IgniterResults,
    output_path: str
) -> str:
    """Generate comprehensive design report with operating envelopes.
    
    Args:
        config: Input configuration
        results: Design results
        output_path: Output HTML filepath
        
    Returns:
        Path to generated HTML file
    """
    generator = ReportGenerator()
    return generator.generate_design_report(
        config, results, output_path, include_envelopes=True
    )
