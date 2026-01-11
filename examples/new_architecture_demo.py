"""
RESA New Architecture Demo

This example demonstrates the new modular architecture with:
- Interface-based design for extensibility
- Plotly-based visualizations
- HTML report generation
- Streamlit-ready components

Run this after implementing the full architecture migration.
"""

# =============================================================================
# EXAMPLE 1: Basic Engine Design with New API
# =============================================================================

def example_basic_usage():
    """
    Demonstrates basic engine design workflow with new architecture.
    """
    # New imports from refactored package
    from resa import LiquidEngine, EngineConfig
    from resa.visualization import EngineDashboardPlotter, CrossSectionPlotter
    from resa.reporting import HTMLReportGenerator

    # 1. Configure Engine
    config = EngineConfig(
        engine_name="Phoenix-2",
        fuel="Ethanol90",
        oxidizer="N2O",
        thrust_n=2200.0,
        pc_bar=25.0,
        mr=4.0,
        L_star=1100.0,
        coolant_name="REFPROP::NitrousOxide",
        cooling_mode="counter-flow",
    )

    # 2. Create Engine (uses default solvers)
    engine = LiquidEngine(config)

    # 3. Run Design (returns pure data, no side effects)
    result = engine.design()

    # 4. Visualize with Plotly
    plotter = EngineDashboardPlotter()
    fig = plotter.create_figure(result)

    # Display interactively
    fig.show()

    # Or get HTML for embedding
    html_plot = plotter.to_html(fig)

    # 5. Generate HTML Report
    reporter = HTMLReportGenerator()
    reporter.generate(result, output_path="output/phoenix2_report.html")

    print(f"Design complete!")
    print(f"  Isp (vac): {result.isp_vac:.1f} s")
    print(f"  Thrust:    {result.thrust_vac:.0f} N")
    print(f"  Report saved to: output/phoenix2_report.html")


# =============================================================================
# EXAMPLE 2: Custom Solver Injection
# =============================================================================

def example_custom_solvers():
    """
    Demonstrates dependency injection for custom/mock solvers.

    This pattern enables:
    - Unit testing with mock solvers
    - Swapping implementations (CEA -> Cantera)
    - Research/experimental solvers
    """
    from resa import LiquidEngine, EngineConfig
    from resa.core.interfaces import Solver
    from dataclasses import dataclass

    # Define a mock combustion result
    @dataclass
    class MockCombustionResult:
        cstar: float = 1500.0
        isp_vac: float = 280.0
        isp_opt: float = 260.0
        T_combustion: float = 3000.0
        gamma: float = 1.2
        Mw: float = 24.0
        expansion_ratio: float = 10.0

    # Create a mock solver for testing
    class MockCEASolver(Solver):
        """Mock combustion solver for testing."""

        def solve(self, pc_bar, mr, eps) -> MockCombustionResult:
            # Return fixed values for testing
            return MockCombustionResult(
                cstar=1500.0 * (pc_bar / 20.0) ** 0.1,
                isp_vac=280.0,
                T_combustion=3000.0,
                gamma=1.2,
            )

        def validate_inputs(self, pc_bar, mr, eps) -> bool:
            return pc_bar > 0 and mr > 0 and eps > 0

    # In the full implementation, you would inject like this:
    # components = EngineComponents(
    #     combustion_solver=MockCEASolver(),
    #     cooling_solver=...,
    #     nozzle_generator=...,
    #     fluid_provider=...,
    # )
    # engine = LiquidEngine(config, components=components)

    print("Custom solver injection example - see code for pattern")


# =============================================================================
# EXAMPLE 3: Plotly Visualization Standalone
# =============================================================================

def example_plotly_visualization():
    """
    Demonstrates standalone Plotly visualization without running engine.

    Useful for:
    - Post-processing saved results
    - Custom analysis visualizations
    - Comparing multiple designs
    """
    from resa.visualization import (
        EngineDashboardPlotter,
        ThrottleCurvePlotter,
        PerformanceContourPlotter,
    )
    from resa.visualization.themes import EngineeringTheme, DarkTheme
    import numpy as np

    # Create throttle curve visualization
    throttle_data = [
        {'pct': 100, 'pc': 25.0, 'mr': 4.0, 'thrust': 2200, 'T_wall_max': 750},
        {'pct': 90, 'pc': 22.5, 'mr': 3.8, 'thrust': 1980, 'T_wall_max': 720},
        {'pct': 80, 'pc': 20.0, 'mr': 3.6, 'thrust': 1760, 'T_wall_max': 690},
        {'pct': 70, 'pc': 17.5, 'mr': 3.4, 'thrust': 1540, 'T_wall_max': 660},
        {'pct': 60, 'pc': 15.0, 'mr': 3.2, 'thrust': 1320, 'T_wall_max': 630},
        {'pct': 50, 'pc': 12.5, 'mr': 3.0, 'thrust': 1100, 'T_wall_max': 600},
    ]

    # Use engineering theme
    plotter = ThrottleCurvePlotter(theme=EngineeringTheme())

    fig = plotter.create_figure(
        throttle_data,
        design_point={'pct': 100, 'pc': 25.0, 'mr': 4.0, 'thrust': 2200}
    )

    # Export to HTML
    html = plotter.to_html(fig, include_plotlyjs='cdn', full_html=True)

    # Save standalone HTML
    with open("output/throttle_curve.html", "w") as f:
        f.write(html)

    print("Throttle curve saved to: output/throttle_curve.html")


# =============================================================================
# EXAMPLE 4: HTML Report with Custom Sections
# =============================================================================

def example_custom_report():
    """
    Demonstrates adding custom sections to HTML reports.
    """
    from resa.reporting import HTMLReportGenerator, ReportSection

    # Create generator
    reporter = HTMLReportGenerator()

    # Add custom analysis section
    reporter.add_section(
        title="Manufacturing Notes",
        content="""
        <p><strong>Liner Material:</strong> CuCrZr alloy (C18150)</p>
        <ul>
            <li>Channel machining: CNC with 0.3mm end mill</li>
            <li>Surface finish: Ra 1.6 μm or better</li>
            <li>Closeout: Electroformed nickel, 1.0mm minimum</li>
        </ul>
        <p><strong>Quality Requirements:</strong></p>
        <ul>
            <li>Hydrostatic test: 1.5x operating pressure</li>
            <li>Flow test: Verify pressure drop within ±10%</li>
            <li>X-ray inspection of braze joints</li>
        </ul>
        """,
        order=95  # Near end of report
    )

    # Add test results section
    reporter.add_section(
        title="Hot Fire Test Results",
        content="""
        <table>
            <tr><th>Test ID</th><th>Duration</th><th>Peak Pc</th><th>Status</th></tr>
            <tr><td>HF-001</td><td>0.5s</td><td>24.8 bar</td><td style="color: green;">PASS</td></tr>
            <tr><td>HF-002</td><td>2.0s</td><td>25.1 bar</td><td style="color: green;">PASS</td></tr>
            <tr><td>HF-003</td><td>5.0s</td><td>25.0 bar</td><td style="color: green;">PASS</td></tr>
        </table>
        """,
        order=90
    )

    # Generate report (would need actual result object)
    # html = reporter.generate(result, output_path="detailed_report.html")

    print("Custom report sections defined - run with actual engine result")


# =============================================================================
# EXAMPLE 5: Streamlit Integration Pattern
# =============================================================================

def example_streamlit_pattern():
    """
    Shows the pattern for Streamlit integration.

    The new architecture makes Streamlit apps clean and simple.
    """
    streamlit_app_code = '''
# ui/app.py - Streamlit Application

import streamlit as st
from resa import LiquidEngine, EngineConfig
from resa.visualization import EngineDashboardPlotter
from resa.reporting import HTMLReportGenerator

st.set_page_config(page_title="RESA Engine Designer", layout="wide")
st.title("RESA Engine Design Suite")

# Sidebar configuration
with st.sidebar:
    st.header("Engine Parameters")
    thrust = st.number_input("Thrust [N]", 100, 10000, 2200)
    pc = st.number_input("Chamber Pressure [bar]", 5.0, 50.0, 25.0)
    mr = st.number_input("Mixture Ratio", 1.0, 8.0, 4.0)

# Create config
config = EngineConfig(
    engine_name="Interactive Design",
    fuel="Ethanol90",
    oxidizer="N2O",
    thrust_n=thrust,
    pc_bar=pc,
    mr=mr,
)

# Run button
if st.button("Run Design", type="primary"):
    engine = LiquidEngine(config)
    result = engine.design()

    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Isp (vac)", f"{result.isp_vac:.1f} s")
    col2.metric("Thrust", f"{result.thrust_vac:.0f} N")
    col3.metric("Mass Flow", f"{result.massflow_total:.3f} kg/s")

    # Display Plotly chart (native Streamlit support!)
    plotter = EngineDashboardPlotter()
    fig = plotter.create_figure(result)
    st.plotly_chart(fig, use_container_width=True)

    # Download report button
    reporter = HTMLReportGenerator()
    html_report = reporter.generate(result)
    st.download_button(
        "Download HTML Report",
        html_report,
        file_name="engine_report.html",
        mime="text/html"
    )
'''

    print("Streamlit integration pattern:")
    print("-" * 60)
    print(streamlit_app_code)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)

    print("=" * 60)
    print("RESA New Architecture Examples")
    print("=" * 60)

    # Note: These examples show the API patterns.
    # Full functionality requires completing the architecture migration.

    print("\n1. Basic Usage Pattern")
    print("-" * 40)
    # example_basic_usage()  # Uncomment when migration is complete

    print("\n2. Custom Solver Injection")
    print("-" * 40)
    example_custom_solvers()

    print("\n3. Plotly Visualization")
    print("-" * 40)
    # example_plotly_visualization()

    print("\n4. Custom Report Sections")
    print("-" * 40)
    example_custom_report()

    print("\n5. Streamlit Integration")
    print("-" * 40)
    example_streamlit_pattern()

    print("\n" + "=" * 60)
    print("See docs/ARCHITECTURE_REFACTOR_PROPOSAL.md for full details")
    print("=" * 60)
