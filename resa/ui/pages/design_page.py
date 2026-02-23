"""
Engine Design Page for RESA UI.

Restructured workflow:
- Tab 1 Configuration: General, Propellants, Nominal Operating Point,
  Efficiencies, Chamber & Nozzle — all cooling inputs removed (live on Cooling page).
- Tab 2 Run & Results: single Run Design button (no cooling solver),
  compact metric grid, nozzle contour + gas dynamics plots, 3D preview, data tables.
- Tab 3 Report: HTML report generation.
- Tab 4 Export: YAML config + geometry CSV download.
"""
import io
import math
import yaml
import numpy as np
import streamlit as st
from datetime import datetime


# ---------------------------------------------------------------------------
# L* recommendations per propellant combination (mm)
# ---------------------------------------------------------------------------
_LSTAR_HINTS = {
    ("Ethanol90", "N2O"): (900, 1300),
    ("Ethanol",   "N2O"): (900, 1300),
    ("RP1",       "LOX"): (1000, 1500),
    ("Methane",   "LOX"): (800, 1200),
    ("LH2",       "LOX"): (700, 1100),
}

_FUEL_DEFAULTS = {
    "Ethanol90": 298.0,
    "Ethanol":   298.0,
    "RP1":       298.0,
    "Methane":   111.0,
    "LH2":       21.0,
    "IPA":       298.0,
    "Propane":   231.0,
}

_OX_DEFAULTS = {
    "N2O": 278.0,
    "LOX": 90.0,
    "H2O2": 298.0,
}

_COOLANT_MAP = {
    "N2O (self-pressurising)": "REFPROP::NitrousOxide",
    "RP-1":    "INCOMP::DowQ",
    "Ethanol": "Ethanol",
    "Water":   "Water",
}


# ---------------------------------------------------------------------------
# Helper: compact section header
# ---------------------------------------------------------------------------
def _section(label: str):
    st.markdown(
        f'<div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.08em;color:#4a8abf;padding:0.6rem 0 0.2rem;">'
        f'{label}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def render_design_page():
    st.markdown(
        '<h2 style="margin-bottom:0.1rem;">Thrust Chamber Design</h2>'
        '<p style="color:#4a6a8a;font-size:0.82rem;margin-top:0;">Combustion · Nozzle Geometry · Gas Dynamics</p>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Configuration", "Run & Results", "Report", "Export"])

    with tab1:
        _render_configuration()
    with tab2:
        _render_run_and_results()
    with tab3:
        _render_report()
    with tab4:
        _render_export()


# ===========================================================================
# TAB 1 — CONFIGURATION
# ===========================================================================

def _render_configuration():
    # ── GENERAL ──────────────────────────────────────────────────────────────
    _section("General")
    g1, g2, g3 = st.columns([3, 1, 2])
    with g1:
        engine_name = st.text_input("Engine Name", value=st.session_state.get("_cfg_name", "Phoenix-1"), key="_cfg_name")
    with g2:
        version = st.text_input("Version", value=st.session_state.get("_cfg_ver", "1.0"), key="_cfg_ver")
    with g3:
        designer = st.text_input("Designer", value=st.session_state.get("_cfg_designer", ""), key="_cfg_designer")
    description = st.text_input(
        "Description",
        value=st.session_state.get("_cfg_desc", ""),
        placeholder="Short description of this design…",
        key="_cfg_desc",
    )

    st.divider()

    # ── PROPELLANTS ───────────────────────────────────────────────────────────
    _section("Propellants")
    p1, p2 = st.columns(2)

    with p1:
        st.markdown('<span style="font-size:0.78rem;color:#7ba7cc;">FUEL</span>', unsafe_allow_html=True)
        fuel_options = ["Ethanol90", "Ethanol", "RP1", "Methane", "LH2", "IPA", "Propane"]
        fuel_idx = fuel_options.index(st.session_state.get("_cfg_fuel", "Ethanol90")) if st.session_state.get("_cfg_fuel", "Ethanol90") in fuel_options else 0
        fuel = st.selectbox("Fuel", fuel_options, index=fuel_idx, key="_cfg_fuel", label_visibility="collapsed")
        fuel_t_default = _FUEL_DEFAULTS.get(fuel, 298.0)
        fuel_t = st.number_input(
            "Injection Temp [K]", min_value=20.0, max_value=500.0,
            value=float(st.session_state.get("_cfg_fuel_t", fuel_t_default)),
            step=1.0, format="%.1f", key="_cfg_fuel_t",
            help="Propellant temperature at injector face",
        )

    with p2:
        st.markdown('<span style="font-size:0.78rem;color:#7ba7cc;">OXIDIZER</span>', unsafe_allow_html=True)
        ox_options = ["N2O", "LOX", "H2O2"]
        ox_idx = ox_options.index(st.session_state.get("_cfg_ox", "N2O")) if st.session_state.get("_cfg_ox", "N2O") in ox_options else 0
        oxidizer = st.selectbox("Oxidizer", ox_options, index=ox_idx, key="_cfg_ox", label_visibility="collapsed")
        ox_t_default = _OX_DEFAULTS.get(oxidizer, 278.0)
        ox_t = st.number_input(
            "Injection Temp [K]", min_value=20.0, max_value=500.0,
            value=float(st.session_state.get("_cfg_ox_t", ox_t_default)),
            step=1.0, format="%.1f", key="_cfg_ox_t",
            help="Propellant temperature at injector face",
        )

    st.divider()

    # ── NOMINAL OPERATING POINT ───────────────────────────────────────────────
    _section("Nominal Operating Point")
    n1, n2, n3 = st.columns(3)
    with n1:
        thrust = st.number_input("Thrust [N]", min_value=10.0, max_value=1e6,
                                  value=float(st.session_state.get("_cfg_thrust", 2200.0)),
                                  step=100.0, format="%.0f", key="_cfg_thrust")
    with n2:
        pc = st.number_input("Chamber Pressure [bar]", min_value=1.0, max_value=300.0,
                              value=float(st.session_state.get("_cfg_pc", 25.0)),
                              step=0.5, format="%.1f", key="_cfg_pc")
    with n3:
        mr = st.number_input("Mixture Ratio O/F", min_value=0.5, max_value=15.0,
                              value=float(st.session_state.get("_cfg_mr", 4.0)),
                              step=0.05, format="%.2f", key="_cfg_mr")

    st.divider()

    # ── EFFICIENCIES ──────────────────────────────────────────────────────────
    _section("Efficiencies")
    e1, e2, e3 = st.columns([2, 2, 2])
    with e1:
        eta_c = st.number_input(
            "Combustion Efficiency η_c*", min_value=0.80, max_value=1.00,
            value=float(st.session_state.get("_cfg_etac", 0.95)),
            step=0.005, format="%.3f", key="_cfg_etac",
            help="c* efficiency: ratio of actual to theoretical characteristic velocity",
        )
    with e2:
        eta_div = st.number_input(
            "Nozzle Divergence Efficiency η_div", min_value=0.90, max_value=1.00,
            value=float(st.session_state.get("_cfg_etadiv", 0.983)),
            step=0.001, format="%.3f", key="_cfg_etadiv",
            help="Accounts for non-axial momentum at nozzle exit. Typical 80% bell ≈ 0.983",
        )
    with e3:
        freeze = st.toggle(
            "Freeze at Throat",
            value=bool(st.session_state.get("_cfg_freeze", False)),
            key="_cfg_freeze",
            help="Assume frozen chemical equilibrium downstream of the throat. "
                 "Conservative for highly dissociating propellants (e.g. H2O2, N2O4).",
        )

    st.divider()

    # ── CHAMBER & NOZZLE ──────────────────────────────────────────────────────
    _section("Chamber & Nozzle")

    # Nozzle type
    ntype_options = ["Bell (TOP)", "Conical", "Ideal (MOC) — coming soon"]
    ntype_stored = st.session_state.get("_cfg_ntype", "Bell (TOP)")
    ntype_idx = ntype_options.index(ntype_stored) if ntype_stored in ntype_options else 0
    ntype = st.radio(
        "Nozzle Type", ntype_options, index=ntype_idx,
        horizontal=True, key="_cfg_ntype",
        help="Bell (TOP) = thrust-optimised parabolic (Rao). Conical = straight wall. Ideal (MOC) = method of characteristics — coming soon.",
    )
    nozzle_type_val = "bell" if ntype.startswith("Bell") else ("conical" if ntype.startswith("Conical") else "ideal")

    # Expansion ratio mode
    exp_options = ["Optimal — Sea Level", "Optimal — Altitude", "Fixed"]
    exp_stored = st.session_state.get("_cfg_expmode", "Optimal — Sea Level")
    exp_idx = exp_options.index(exp_stored) if exp_stored in exp_options else 0
    exp_mode = st.radio(
        "Expansion Ratio", exp_options, index=exp_idx,
        horizontal=True, key="_cfg_expmode",
    )

    exp1, exp2 = st.columns(2)
    with exp1:
        if exp_mode == "Fixed":
            eps = st.number_input("Expansion Ratio ε", min_value=1.5, max_value=200.0,
                                   value=float(st.session_state.get("_cfg_eps", 6.0)),
                                   step=0.1, format="%.1f", key="_cfg_eps")
            p_exit = 1.013
        elif exp_mode == "Optimal — Altitude":
            p_exit = st.number_input("Design Ambient Pressure [bar]", min_value=0.001, max_value=1.013,
                                      value=float(st.session_state.get("_cfg_pexit", 0.1)),
                                      step=0.005, format="%.4f", key="_cfg_pexit")
            eps = 0.0
        else:
            p_exit = 1.013
            eps = 0.0
            st.markdown('<span style="font-size:0.8rem;color:#4a6a8a;">Expansion ratio optimised for sea-level (p_exit = 1.013 bar)</span>', unsafe_allow_html=True)
    with exp2:
        pass  # reserved for future computed display

    # Geometry row
    h1, h2, h3 = st.columns(3)
    lstar_hint = _LSTAR_HINTS.get((fuel, oxidizer), (800, 1400))
    with h1:
        L_star = st.number_input(
            "L* [mm]", min_value=300.0, max_value=3000.0,
            value=float(st.session_state.get("_cfg_lstar", 1100.0)),
            step=50.0, format="%.0f", key="_cfg_lstar",
            help=f"Characteristic chamber length. Typical for {fuel}/{oxidizer}: {lstar_hint[0]}–{lstar_hint[1]} mm",
        )
        st.caption(f"Typical: {lstar_hint[0]}–{lstar_hint[1]} mm")
    with h2:
        cr = st.number_input(
            "Contraction Ratio", min_value=2.0, max_value=20.0,
            value=float(st.session_state.get("_cfg_cr", 8.0)),
            step=0.5, format="%.1f", key="_cfg_cr",
            help="Chamber area / throat area. Typical: 5–15",
        )
        st.caption("Typical: 5–15")
    with h3:
        theta_conv = st.number_input(
            "Convergent Half-Angle [°]", min_value=15.0, max_value=60.0,
            value=float(st.session_state.get("_cfg_tconv", 30.0)),
            step=1.0, format="%.1f", key="_cfg_tconv",
            help="Half-angle of the convergent cone. Typical: 25–45°",
        )
        st.caption("Typical: 25–45°")

    # Bell-specific / conical-specific second row
    if nozzle_type_val == "bell":
        b1, b2 = st.columns([2, 4])
        with b1:
            bell_frac = st.slider(
                "Bell Fraction", min_value=0.60, max_value=1.00,
                value=float(st.session_state.get("_cfg_bellfrac", 0.80)),
                step=0.01, format="%.2f", key="_cfg_bellfrac",
                help="Fraction of the 15° conical equivalent length. 0.80 is industry standard.",
            )
        with b2:
            st.markdown("")
    elif nozzle_type_val == "conical":
        bell_frac = 0.8  # unused but keep default
        c1, _ = st.columns([2, 4])
        with c1:
            theta_exit_val = st.number_input(
                "Exit Half-Angle [°]", min_value=5.0, max_value=30.0,
                value=float(st.session_state.get("_cfg_texit", 15.0)),
                step=0.5, format="%.1f", key="_cfg_texit",
                help="Conical nozzle divergence half-angle. Typical: 12–18°",
            )
        st.caption("Typical: 12–18° for conical")
    else:
        bell_frac = 0.8
        st.info("Ideal (MOC) nozzle contour generation is coming soon.", icon="🔧")

    theta_exit_val = st.session_state.get("_cfg_texit", 15.0) if nozzle_type_val == "conical" else 15.0

    st.divider()

    # ── SAVE ─────────────────────────────────────────────────────────────────
    sb1, sb2 = st.columns([2, 1])
    with sb1:
        save_clicked = st.button("Save Configuration", type="primary", use_container_width=True)
    with sb2:
        export_yaml_now = st.button("Export YAML", use_container_width=True,
                                     disabled=st.session_state.get("engine_config") is None,
                                     help="Save config to YAML file (requires a saved config)")

    if save_clicked:
        try:
            from resa.core.config import EngineConfig

            config = EngineConfig(
                engine_name=engine_name,
                version=version,
                designer=designer,
                description=description,
                fuel=fuel,
                oxidizer=oxidizer,
                fuel_injection_temp_k=fuel_t,
                oxidizer_injection_temp_k=ox_t,
                thrust_n=thrust,
                pc_bar=pc,
                mr=mr,
                eff_combustion=eta_c,
                eff_nozzle_divergence=eta_div,
                freeze_at_throat=freeze,
                nozzle_type=nozzle_type_val,
                expansion_ratio=eps,
                p_exit_bar=p_exit,
                L_star=L_star,
                contraction_ratio=cr,
                theta_convergent=theta_conv,
                theta_exit=float(theta_exit_val),
                bell_fraction=bell_frac if nozzle_type_val == "bell" else 0.8,
            )

            validation = config.validate()
            st.session_state.engine_config = config

            if validation.errors:
                for e in validation.errors:
                    st.error(e)
            elif validation.warnings:
                for w in validation.warnings:
                    st.warning(w)
                st.success(f"Configuration saved: **{engine_name}**  (with warnings above)")
            else:
                st.success(f"Configuration saved: **{engine_name}**")

            if "analysis_history" not in st.session_state:
                st.session_state.analysis_history = []
            st.session_state.analysis_history.append(
                f"{datetime.now().strftime('%H:%M')} — Config saved: {engine_name} "
                f"({fuel}/{oxidizer}, {thrust:.0f} N, {pc:.1f} bar)"
            )

        except Exception as exc:
            st.error(f"Error creating config: {exc}")

    if export_yaml_now and st.session_state.get("engine_config"):
        _offer_yaml_download(st.session_state.engine_config)


# ===========================================================================
# TAB 2 — RUN & RESULTS
# ===========================================================================

def _render_run_and_results():
    if not st.session_state.get("engine_config"):
        st.info("Configure and save the engine first (Configuration tab).")
        return

    config = st.session_state.engine_config

    # ── Config summary strip ─────────────────────────────────────────────────
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric("Engine", config.engine_name)
    with s2:
        st.metric("Thrust Target", f"{config.thrust_n:.0f} N")
    with s3:
        st.metric("Pc", f"{config.pc_bar:.1f} bar")
    with s4:
        st.metric("O/F", f"{config.mr:.2f}")
    with s5:
        st.metric("Propellants", f"{config.fuel} / {config.oxidizer}")

    st.divider()

    run_col, _ = st.columns([2, 5])
    with run_col:
        run_clicked = st.button("Run Design", type="primary", use_container_width=True,
                                 help="Runs combustion analysis + nozzle geometry (no cooling solver). "
                                      "Go to Cooling Analysis for thermal simulation.")

    if run_clicked:
        _run_design(config)

    result = st.session_state.get("design_result")
    if result is None:
        return

    _render_results(result, config)


def _run_design(config):
    with st.spinner("Running design…"):
        prog = st.progress(0)
        try:
            from resa.core.engine import Engine

            prog.progress(10, "Initialising solvers…")
            engine = Engine(config)
            prog.progress(25, "Running CEA combustion…")
            prog.progress(60, "Generating nozzle geometry…")
            prog.progress(80, "Analysing gas dynamics…")

            result = engine.design(with_cooling=False)

            prog.progress(100, "Complete")
            st.session_state.design_result = result

            if "analysis_history" not in st.session_state:
                st.session_state.analysis_history = []
            st.session_state.analysis_history.append(
                f"{datetime.now().strftime('%H:%M')} — Design: {config.engine_name}, "
                f"Isp={result.isp_vac:.1f} s, Dt={result.dt_mm:.2f} mm"
            )
            st.success("Design complete.")
        except Exception as exc:
            import traceback
            st.error(f"Design failed: {exc}")
            st.code(traceback.format_exc())


def _render_results(result, config):
    from resa.core.results import CombustionResult

    comb: CombustionResult = result.combustion
    geo = result.nozzle_geometry

    # ── Key computed quantities ───────────────────────────────────────────────
    g0 = 9.80665
    cf_vac = result.thrust_vac / (result.massflow_total * comb.cstar * config.eff_combustion) if comb.cstar > 0 else 0.0
    # Effective cstar (real)
    cstar_real = comb.cstar * config.eff_combustion

    # Chamber dimensions
    dc_mm = (geo.chamber_radius * 2 * 1000) if geo.chamber_radius > 0 else result.dt_mm * math.sqrt(config.contraction_ratio)
    # Chamber length from L* and throat area
    at_m2 = math.pi * (result.dt_mm / 2000) ** 2
    lc_mm = (config.L_star * 1e-3 / at_m2) * at_m2 * 1000  # = L* directly in mm (L* ≡ Vc/At, Lc ≈ L*/(CR-1)*CR... simplified)
    # Better: Lc = (L_star_mm - nozzle_convergent_length) — use L* directly as displayed
    # Throat area and exit area
    at_mm2 = math.pi * (result.dt_mm / 2) ** 2
    ae_mm2 = at_mm2 * result.expansion_ratio

    # Exit angle from geometry (if available)
    theta_exit_deg = math.degrees(geo.theta_exit) if geo.theta_exit else 0.0

    st.divider()

    # ── Performance metrics ───────────────────────────────────────────────────
    _section("Performance")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Isp vac", f"{result.isp_vac:.1f} s")
    with m2:
        st.metric("Isp SL", f"{result.isp_sea:.1f} s")
    with m3:
        st.metric("CF vac", f"{cf_vac:.4f}")
    with m4:
        st.metric("c*", f"{cstar_real:.0f} m/s")
    with m5:
        st.metric("Thrust vac", f"{result.thrust_vac:.0f} N")
    with m6:
        st.metric("ṁ total", f"{result.massflow_total:.4f} kg/s")

    # ── Geometry metrics ──────────────────────────────────────────────────────
    _section("Key Dimensions")
    g1, g2, g3, g4, g5, g6 = st.columns(6)
    with g1:
        st.metric("Throat Ø", f"{result.dt_mm:.2f} mm")
    with g2:
        st.metric("Exit Ø", f"{result.de_mm:.2f} mm")
    with g3:
        st.metric("Chamber Ø", f"{dc_mm:.1f} mm")
    with g4:
        st.metric("L*", f"{config.L_star:.0f} mm")
    with g5:
        st.metric("Expansion ε", f"{result.expansion_ratio:.2f}")
    with g6:
        st.metric("Exit Angle", f"{theta_exit_deg:.1f}°" if theta_exit_deg > 0 else "—")

    st.divider()

    # ── Plots: Nozzle Contour + Gas Dynamics ─────────────────────────────────
    _section("Geometry & Flow")
    col_l, col_r = st.columns(2)

    with col_l:
        try:
            from resa.visualization.engine_plots import NozzleContourPlotter
            from resa.visualization.themes import DarkTheme
            plotter = NozzleContourPlotter(theme=DarkTheme())
            fig = plotter.create_figure(result)
            st.plotly_chart(fig, use_container_width=True, theme=None)
        except Exception as exc:
            st.error(f"Nozzle contour plot error: {exc}")

    with col_r:
        try:
            from resa.visualization.engine_plots import GasDynamicsPlotter
            from resa.visualization.themes import DarkTheme
            plotter = GasDynamicsPlotter(theme=DarkTheme())
            fig = plotter.create_figure(result)
            st.plotly_chart(fig, use_container_width=True, theme=None)
        except Exception as exc:
            st.error(f"Gas dynamics plot error: {exc}")

    st.divider()

    # ── 3D Chamber Preview ────────────────────────────────────────────────────
    _section("3D Chamber Preview (inner wall)")
    try:
        from resa.visualization.engine_3d import Engine3DViewer
        from resa.visualization.themes import DarkTheme
        viewer = Engine3DViewer(theme=DarkTheme(), dark_mode=True)
        fig3d = viewer.render_nozzle(result.nozzle_geometry)
        st.plotly_chart(fig3d, use_container_width=True, theme=None)
    except Exception as exc:
        st.error(f"3D view error: {exc}")

    st.divider()

    # ── Data Tables ───────────────────────────────────────────────────────────
    _section("Detailed Results")

    with st.expander("Performance", expanded=False):
        import pandas as pd
        perf_data = {
            "Parameter": [
                "Isp (vacuum)", "Isp (sea level)",
                "Thrust (vacuum)", "Thrust (sea level)",
                "Thrust coefficient CF (vac)",
                "Characteristic velocity c*",
                "Total mass flow",
                "Oxidiser flow", "Fuel flow",
                "O/F ratio",
            ],
            "Value": [
                f"{result.isp_vac:.2f} s", f"{result.isp_sea:.2f} s",
                f"{result.thrust_vac:.2f} N", f"{result.thrust_sea:.2f} N",
                f"{cf_vac:.4f}",
                f"{cstar_real:.2f} m/s",
                f"{result.massflow_total:.5f} kg/s",
                f"{result.massflow_ox:.5f} kg/s",
                f"{result.massflow_fuel:.5f} kg/s",
                f"{result.mr:.3f}",
            ],
        }
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    with st.expander("Geometry", expanded=False):
        import pandas as pd
        geo_data = {
            "Parameter": [
                "Throat diameter", "Exit diameter", "Chamber diameter",
                "Throat area", "Exit area",
                "Expansion ratio ε", "Contraction ratio",
                "Characteristic length L*",
                "Total nozzle length",
                "Exit half-angle",
            ],
            "Value": [
                f"{result.dt_mm:.3f} mm", f"{result.de_mm:.3f} mm", f"{dc_mm:.2f} mm",
                f"{at_mm2:.4f} mm²", f"{ae_mm2:.2f} mm²",
                f"{result.expansion_ratio:.3f}", f"{config.contraction_ratio:.2f}",
                f"{config.L_star:.0f} mm",
                f"{result.length_mm:.1f} mm",
                f"{theta_exit_deg:.2f}°" if theta_exit_deg > 0 else "—",
            ],
        }
        st.dataframe(pd.DataFrame(geo_data), use_container_width=True, hide_index=True)

    with st.expander("Thermodynamics (CEA)", expanded=False):
        import pandas as pd
        # Isentropic relations at throat and exit
        gamma = comb.gamma
        t_throat = comb.T_combustion * (2 / (gamma + 1))
        pc_pa = config.pc_bar * 1e5
        pt_pa = pc_pa * (2 / (gamma + 1)) ** (gamma / (gamma - 1))
        thermo_data = {
            "Parameter": [
                "Chamber temperature Tc", "Throat temperature Tt",
                "Chamber pressure Pc", "Throat pressure Pt",
                "Specific heat ratio γ",
                "Molecular weight MW",
                "Theoretical c*",
                "η_c* (combustion eff.)",
                "η_div (divergence eff.)",
                "Freeze at throat",
            ],
            "Value": [
                f"{comb.T_combustion:.0f} K", f"{t_throat:.0f} K",
                f"{config.pc_bar:.2f} bar", f"{pt_pa/1e5:.3f} bar",
                f"{gamma:.4f}",
                f"{comb.mw:.2f} g/mol",
                f"{comb.cstar:.2f} m/s",
                f"{config.eff_combustion:.3f}",
                f"{config.eff_nozzle_divergence:.3f}",
                "Yes" if config.freeze_at_throat else "No",
            ],
        }
        st.dataframe(pd.DataFrame(thermo_data), use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 3 — REPORT
# ===========================================================================

def _render_report():
    st.subheader("Generate Report")

    if not st.session_state.get("design_result"):
        st.info("Run the design analysis first (Run & Results tab).")
        return

    result = st.session_state.design_result
    config = st.session_state.engine_config

    c1, c2 = st.columns(2)
    with c1:
        include_plots = st.checkbox("Performance plots", value=True)
        include_geometry = st.checkbox("Geometry details", value=True)
    with c2:
        include_3d = st.checkbox("3D visualisation", value=False)
        include_raw = st.checkbox("Raw data tables", value=False)

    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating…"):
            try:
                from resa.reporting.html_report import HTMLReportGenerator
                generator = HTMLReportGenerator()
                html = generator.generate(result)
                st.success("Report ready.")
                st.download_button(
                    "Download HTML Report",
                    html,
                    file_name=f"{config.engine_name}_report.html",
                    mime="text/html",
                )
                with st.expander("Preview"):
                    st.components.v1.html(html, height=800, scrolling=True)
            except Exception as exc:
                st.error(f"Report generation failed: {exc}")


# ===========================================================================
# TAB 4 — EXPORT
# ===========================================================================

def _render_export():
    st.subheader("Export")

    config = st.session_state.get("engine_config")
    result = st.session_state.get("design_result")

    # YAML
    st.markdown("#### Configuration (YAML)")
    if config:
        _offer_yaml_download(config)
    else:
        st.info("Save a configuration first.")

    st.divider()

    # Geometry CSV
    st.markdown("#### Nozzle Contour (CSV)")
    if result and result.nozzle_geometry is not None:
        geo = result.nozzle_geometry
        x_mm = geo.x_full * 1000
        y_mm = geo.y_full * 1000
        at = math.pi * geo.throat_radius ** 2 if geo.throat_radius > 0 else (math.pi * (result.dt_mm / 2000) ** 2)
        area_ratio = (math.pi * (y_mm / 1000) ** 2) / at

        import pandas as pd
        df = pd.DataFrame({
            "x_mm": x_mm,
            "r_mm": y_mm,
            "area_ratio": area_ratio,
        })
        csv = df.to_csv(index=False)
        st.download_button(
            "Download Contour CSV",
            csv,
            file_name=f"{config.engine_name if config else 'engine'}_contour.csv",
            mime="text/csv",
        )
        st.caption(f"{len(x_mm)} axial stations · throat at x = {x_mm[np.argmin(y_mm)]:.1f} mm")
    else:
        st.info("Run a design first to export geometry.")


# ===========================================================================
# HELPERS
# ===========================================================================

def _offer_yaml_download(config):
    """Render a download button for config as YAML."""
    import yaml as _yaml
    buf = io.StringIO()
    _yaml.dump(config.to_dict(), buf, default_flow_style=False, sort_keys=False)
    st.download_button(
        "Download Config YAML",
        buf.getvalue(),
        file_name=f"{config.engine_name}_config.yaml",
        mime="text/yaml",
    )
