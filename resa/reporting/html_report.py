"""
HTML Report Generator for RESA Engine Analysis.

Creates comprehensive, self-contained HTML reports with:
- Embedded interactive Plotly charts
- Performance metrics summary
- Configuration details
- Cooling system analysis
- Exportable to PDF via browser print
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime
import json
import numpy as np

if TYPE_CHECKING:
    from resa.core.results import EngineDesignResult


@dataclass
class ReportSection:
    """
    Represents a custom section in the report.

    Attributes:
        title: Section heading
        content: HTML content for the section
        order: Position in report (lower = earlier)
    """
    title: str
    content: str
    order: int = 100


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    include_dashboard: bool = True
    include_cross_section: bool = True
    include_3d: bool = False
    include_raw_data: bool = False
    company_name: Optional[str] = None
    company_logo_url: Optional[str] = None
    custom_css: Optional[str] = None


class HTMLReportGenerator:
    """
    Generates comprehensive HTML reports from engine analysis results.

    Features:
    - Embedded interactive Plotly charts (using CDN)
    - Responsive design works on all devices
    - Print-friendly styling
    - Customizable sections and theming

    Example:
        from resa.reporting import HTMLReportGenerator

        generator = HTMLReportGenerator()
        html = generator.generate(engine_result)

        # Save to file
        with open("report.html", "w") as f:
            f.write(html)

        # Or generate directly to file
        generator.generate(engine_result, output_path="report.html")
    """

    def __init__(
        self,
        config: Optional[ReportConfig] = None,
        theme: Optional[Any] = None
    ):
        """
        Initialize the report generator.

        Args:
            config: Report configuration options
            theme: PlotTheme for consistent visualization styling
        """
        self.config = config or ReportConfig()
        self.theme = theme
        self._custom_sections: List[ReportSection] = []

        # Lazy import plotters to avoid circular imports
        self._dashboard_plotter = None
        self._cross_section_plotter = None

    def _get_plotters(self):
        """Lazy initialization of plotters."""
        if self._dashboard_plotter is None:
            from resa.visualization.engine_plots import (
                EngineDashboardPlotter,
                CrossSectionPlotter
            )
            self._dashboard_plotter = EngineDashboardPlotter(self.theme)
            self._cross_section_plotter = CrossSectionPlotter(self.theme)

    def add_section(self, title: str, content: str, order: int = 100) -> None:
        """
        Add a custom section to the report.

        Args:
            title: Section heading
            content: HTML content
            order: Position (lower = earlier, default sections are 10-90)
        """
        self._custom_sections.append(ReportSection(title, content, order))

    def generate(
        self,
        result: 'EngineDesignResult',
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> str:
        """
        Generate HTML report from engine results.

        Args:
            result: EngineDesignResult from engine.design() or analyze()
            output_path: If provided, save HTML to this file
            title: Custom report title

        Returns:
            Complete HTML document as string
        """
        self._get_plotters()

        # Extract engine name
        engine_name = "Engine"
        if hasattr(result, 'config') and hasattr(result.config, 'engine_name'):
            engine_name = result.config.engine_name

        # Generate embedded plots
        plots_html = self._generate_plots(result)

        # Build report context
        context = {
            'title': title or f"Engine Design Report: {engine_name}",
            'engine_name': engine_name,
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': self._format_config(result),
            'performance': self._format_performance(result),
            'geometry': self._format_geometry(result),
            'cooling': self._format_cooling(result),
            'plots': plots_html,
            'custom_sections': sorted(self._custom_sections, key=lambda s: s.order),
            'report_config': self.config,
        }

        # Render HTML
        html = self._render_template(context)

        # Save if path provided
        if output_path:
            Path(output_path).write_text(html, encoding='utf-8')

        return html

    def _generate_plots(self, result: 'EngineDesignResult') -> Dict[str, str]:
        """Generate all plots as embeddable HTML."""
        plots = {}

        # Main dashboard
        if self.config.include_dashboard:
            dashboard_fig = self._dashboard_plotter.create_figure(result)
            plots['dashboard'] = dashboard_fig.to_html(
                include_plotlyjs=False,  # Will include once in head
                full_html=False,
                config={'responsive': True}
            )

        # Cross-section at throat
        if self.config.include_cross_section:
            throat_idx = np.argmin(result.geometry.y_full)
            cross_fig = self._cross_section_plotter.create_figure(
                result.channel_geometry,
                station_idx=throat_idx,
                sector_angle=90
            )
            plots['cross_section'] = cross_fig.to_html(
                include_plotlyjs=False,
                full_html=False,
                config={'responsive': True}
            )

        return plots

    def _format_config(self, result: 'EngineDesignResult') -> Dict[str, str]:
        """Format configuration for display."""
        if not hasattr(result, 'config'):
            return {}

        cfg = result.config
        return {
            'Fuel': str(cfg.fuel),
            'Oxidizer': str(cfg.oxidizer),
            'Chamber Pressure': f"{cfg.pc_bar:.1f} bar",
            'Mixture Ratio': f"{cfg.mr:.2f}",
            'Target Thrust': f"{cfg.thrust_n:.0f} N",
            'L*': f"{cfg.L_star:.0f} mm",
            'Coolant': cfg.coolant_name.split('::')[-1] if '::' in cfg.coolant_name else cfg.coolant_name,
            'Cooling Mode': cfg.cooling_mode,
        }

    def _format_performance(self, result: 'EngineDesignResult') -> Dict[str, str]:
        """Format performance metrics for display."""
        return {
            'isp_vac': f"{result.isp_vac:.1f}",
            'isp_sea': f"{result.isp_sea:.1f}",
            'thrust_vac': f"{result.thrust_vac:.0f}",
            'thrust_sea': f"{result.thrust_sea:.0f}",
            'massflow': f"{result.massflow_total:.4f}",
            'expansion_ratio': f"{result.expansion_ratio:.2f}",
        }

    def _format_geometry(self, result: 'EngineDesignResult') -> Dict[str, str]:
        """Format geometry data for display."""
        return {
            'throat_diameter': f"{result.dt_mm:.2f}",
            'exit_diameter': f"{result.de_mm:.2f}",
            'length': f"{result.length_mm:.1f}",
        }

    def _format_cooling(self, result: 'EngineDesignResult') -> Dict[str, str]:
        """Format cooling summary for display."""
        cooling = result.cooling_data
        return {
            'max_wall_temp': f"{np.max(cooling['T_wall_hot']):.0f}",
            'max_coolant_temp': f"{np.max(cooling['T_coolant']):.0f}",
            'max_heat_flux': f"{np.max(cooling['q_flux']) / 1e6:.2f}",
            'pressure_drop': f"{(np.max(cooling['P_coolant']) - np.min(cooling['P_coolant'])) / 1e5:.2f}",
            'min_density': f"{np.min(cooling['density']):.1f}",
        }

    def _render_template(self, context: Dict[str, Any]) -> str:
        """Render the HTML template with context."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{context['title']}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --primary: #2c3e50;
            --secondary: #3498db;
            --accent: #e74c3c;
            --success: #27ae60;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
            --text: #333333;
            --text-light: #666666;
            --border: #e0e0e0;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif;
            line-height: 1.6;
            color: var(--text);
            background: var(--bg);
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        /* Header */
        header {{
            background: linear-gradient(135deg, var(--primary) 0%, #1a252f 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}

        header h1 {{
            font-size: 2.2em;
            margin-bottom: 10px;
            font-weight: 600;
        }}

        header .meta {{
            opacity: 0.85;
            font-size: 0.95em;
        }}

        /* Cards */
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            padding: 28px;
            margin-bottom: 25px;
            border: 1px solid var(--border);
        }}

        .card h2 {{
            color: var(--primary);
            font-size: 1.4em;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 3px solid var(--secondary);
            font-weight: 600;
        }}

        /* Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
        }}

        .metric {{
            text-align: center;
            padding: 25px 15px;
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border-radius: 10px;
            border: 1px solid var(--border);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .metric:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        .metric-value {{
            font-size: 2.2em;
            font-weight: 700;
            color: var(--primary);
            line-height: 1.2;
        }}

        .metric-label {{
            color: var(--text-light);
            font-size: 0.9em;
            margin-top: 8px;
        }}

        .metric-unit {{
            font-size: 0.6em;
            font-weight: 400;
            opacity: 0.7;
        }}

        /* Two Column Layout */
        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }}

        @media (max-width: 900px) {{
            .two-column {{
                grid-template-columns: 1fr;
            }}
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}

        th, td {{
            padding: 14px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            background: var(--bg);
            font-weight: 600;
            color: var(--primary);
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        tr:hover {{
            background: #f8f9fa;
        }}

        /* Plot Container */
        .plot-container {{
            width: 100%;
            min-height: 400px;
            margin: 20px 0;
        }}

        .plot-container-small {{
            max-width: 700px;
            margin: 20px auto;
        }}

        /* Cooling Status */
        .status-good {{
            color: var(--success);
            font-weight: 600;
        }}

        .status-warning {{
            color: #f39c12;
            font-weight: 600;
        }}

        .status-danger {{
            color: var(--accent);
            font-weight: 600;
        }}

        /* Footer */
        footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-light);
            font-size: 0.9em;
            border-top: 1px solid var(--border);
            margin-top: 40px;
        }}

        /* Print Styles */
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}

            .card {{
                break-inside: avoid;
                box-shadow: none;
                border: 1px solid #ddd;
            }}

            header {{
                background: var(--primary) !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}

            .no-print {{
                display: none;
            }}
        }}

        {context['report_config'].custom_css or ''}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <h1>{context['engine_name']}</h1>
            <p class="meta">
                Design Report | Generated: {context['generated_at']}
            </p>
        </header>

        <!-- Performance Summary -->
        <div class="card">
            <h2>Performance Summary</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{context['performance']['isp_vac']}<span class="metric-unit"> s</span></div>
                    <div class="metric-label">Vacuum Isp</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{context['performance']['thrust_vac']}<span class="metric-unit"> N</span></div>
                    <div class="metric-label">Vacuum Thrust</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{context['performance']['massflow']}<span class="metric-unit"> kg/s</span></div>
                    <div class="metric-label">Mass Flow</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{context['performance']['expansion_ratio']}</div>
                    <div class="metric-label">Expansion Ratio</div>
                </div>
            </div>
        </div>

        <!-- Configuration & Geometry -->
        <div class="two-column">
            <div class="card">
                <h2>Configuration</h2>
                <table>
                    {''.join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in context['config'].items())}
                </table>
            </div>

            <div class="card">
                <h2>Geometry & Cooling</h2>
                <table>
                    <tr><th>Throat Diameter</th><td>{context['geometry']['throat_diameter']} mm</td></tr>
                    <tr><th>Exit Diameter</th><td>{context['geometry']['exit_diameter']} mm</td></tr>
                    <tr><th>Chamber Length</th><td>{context['geometry']['length']} mm</td></tr>
                </table>

                <h3 style="margin-top: 25px; margin-bottom: 15px; font-size: 1.1em; color: var(--primary);">Cooling Analysis</h3>
                <table>
                    <tr>
                        <th>Max Wall Temp</th>
                        <td class="{self._get_temp_status_class(float(context['cooling']['max_wall_temp']))}">{context['cooling']['max_wall_temp']} K</td>
                    </tr>
                    <tr><th>Max Coolant Temp</th><td>{context['cooling']['max_coolant_temp']} K</td></tr>
                    <tr><th>Max Heat Flux</th><td>{context['cooling']['max_heat_flux']} MW/m²</td></tr>
                    <tr><th>Pressure Drop</th><td>{context['cooling']['pressure_drop']} bar</td></tr>
                    <tr><th>Min Coolant Density</th><td>{context['cooling']['min_density']} kg/m³</td></tr>
                </table>
            </div>
        </div>

        <!-- Main Dashboard Plot -->
        {'<div class="card"><h2>Thermal & Flow Analysis</h2><div class="plot-container">' + context['plots'].get('dashboard', '') + '</div></div>' if 'dashboard' in context['plots'] else ''}

        <!-- Cross Section -->
        {'<div class="card"><h2>Channel Cross-Section (Throat)</h2><div class="plot-container plot-container-small">' + context['plots'].get('cross_section', '') + '</div></div>' if 'cross_section' in context['plots'] else ''}

        <!-- Custom Sections -->
        {''.join(f'<div class="card"><h2>{s.title}</h2>{s.content}</div>' for s in context['custom_sections'])}

        <!-- Footer -->
        <footer>
            <p>Generated by RESA (Rocket Engine Sizing & Analysis) v2.0</p>
            <p>Report timestamp: {context['generated_at']}</p>
        </footer>
    </div>
</body>
</html>'''

    def _get_temp_status_class(self, temp: float) -> str:
        """Determine CSS class based on wall temperature."""
        if temp < 700:
            return "status-good"
        elif temp < 850:
            return "status-warning"
        else:
            return "status-danger"
