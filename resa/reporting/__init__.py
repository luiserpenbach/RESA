"""
RESA Reporting Module

Generates comprehensive HTML reports with embedded Plotly visualizations.

Features:
- Interactive charts embedded in HTML
- Responsive design for web and print
- Customizable templates
- Optional PDF export
"""

from resa.reporting.html_report import HTMLReportGenerator, ReportSection

__all__ = [
    "HTMLReportGenerator",
    "ReportSection",
]
