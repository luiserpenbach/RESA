"""
RESA — Regenerative Cooling Analysis Page
==========================================

Comprehensive two-phase N2O cooling analysis with:
- Bartz hot-gas heat flux model
- Gnielinski / Dittus-Boelter single-phase convection
- Chen saturated boiling correlation
- Bergles-Rohsenow subcooled boiling / ONB criterion
- Bowring Critical Heat Flux with 2x safety margin enforcement
- Jackson supercritical HT with HTD warning
- Lockhart-Martinelli two-phase pressure drop
- 1D marching solver with full phase-aware regime switching
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

_root = Path(__file__).parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    from resa.physics.cooling_n2o import (
        N2OCoolingSolver,
        CoolingChannelGeometry,
        CoolingAnalysisResult,
        N2OConstants,
        FlowRegime,
        get_saturation_properties,
        HeatTransferCorrelations,
    )
    N2O_OK = True
except ImportError:
    N2O_OK = False

try:
    from resa.visualization.themes import DarkTheme
    _THEME = DarkTheme()
except ImportError:
    _THEME = None


# ── Regime colour / label maps ─────────────────────────────────────────────
_REGIME_COLOR = {
    "subcooled_liquid":       "#4a9eff",
    "subcooled_boiling":      "#00e5ff",
    "saturated_boiling":      "#2ecc71",
    "annular_flow":           "#f0c040",
    "mist_flow":              "#ff9d3a",
    "chf_risk":               "#ff4d4d",
    "post_chf":               "#cc0000",
    "superheated_vapor":      "#ff6b6b",
    "supercritical":          "#c084fc",
    "pseudo_critical":        "#f472b6",
    "supercritical_gas_like": "#a78bfa",
}
_REGIME_LABEL = {
    "subcooled_liquid":       "Subcooled Liquid",
    "subcooled_boiling":      "Subcooled Boiling",
    "saturated_boiling":      "Saturated Boiling",
    "annular_flow":           "Annular Flow",
    "mist_flow":              "Mist Flow",
    "chf_risk":               "CHF Risk",
    "post_chf":               "Post-CHF ⚠",
    "superheated_vapor":      "Superheated Vapour",
    "supercritical":          "Supercritical",
    "pseudo_critical":        "Near Pseudo-critical",
    "supercritical_gas_like": "SC Gas-like",
}


# ── UI helpers ─────────────────────────────────────────────────────────────

def _apply_theme(fig):
    if _THEME:
        _THEME.apply_to_figure(fig)
    return fig


def _section(title: str, icon: str = ""):
    st.markdown(
        f'<div style="border-left:3px solid #2e6fff;padding-left:0.8rem;'
        f'margin:1.2rem 0 0.6rem;">'
        f'<span style="font-size:1rem;font-weight:700;color:#e8f4fd;">'
        f'{icon} {title}</span></div>',
        unsafe_allow_html=True,
    )


def _kpi(label, value, sub="", color="#4a9eff"):
    st.markdown(
        f'<div style="background:#111827;border:1px solid #1f2d45;border-radius:10px;'
        f'padding:0.9rem 1.1rem;">'
        f'<div style="font-size:0.68rem;color:#4a6a8a;text-transform:uppercase;'
        f'letter-spacing:0.06em;">{label}</div>'
        f'<div style="font-size:1.4rem;font-weight:700;color:{color};">{value}</div>'
        f'<div style="font-size:0.72rem;color:#4a6a8a;">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _badge(text, kind="info"):
    c = {"safe":("#0a2618","#21c97a","#21c97a"),
         "warn":("#231a06","#f0a000","#f0a000"),
         "danger":("#1e0a0a","#d64045","#d64045"),
         "info":("#091a2e","#4a9eff","#4a9eff")}
    bg, fg, bd = c.get(kind, c["info"])
    st.markdown(
        f'<span style="display:inline-block;background:{bg};color:{fg};'
        f'border:1px solid {bd};border-radius:20px;padding:3px 14px;'
        f'font-size:0.78rem;font-weight:600;">{text}</span>',
        unsafe_allow_html=True,
    )


# ── Model documentation panel ──────────────────────────────────────────────

def _render_model_docs():
    with st.expander("Physical Models & Assumptions  —  click to expand", expanded=False):
        t = st.tabs(["Hot-Gas Side","Single-Phase HT","Boiling HT",
                     "Critical Heat Flux","Supercritical","Pressure Drop",
                     "Fluid Properties","Key Assumptions"])

        with t[0]:
            st.markdown(r"""
### Bartz Equation — Hot-Gas Convection

$$h_g = \frac{0.026}{D_t^{0.2}} \left(\frac{\mu^{0.2} c_p}{Pr^{0.6}}\right)
       \left(\frac{\dot{m}}{A_t}\right)^{0.8}
       \left(\frac{A_t}{A}\right)^{0.9} \sigma$$

Boundary-layer correction factor:

$$\sigma = \left[\tfrac{1}{2}\frac{T_w}{T_c}\left(1+\frac{\gamma-1}{2}Ma^2\right)+\tfrac{1}{2}\right]^{-0.68}
           \!\left[1+\frac{\gamma-1}{2}Ma^2\right]^{-0.12}$$

**Recovery (adiabatic wall) temperature:**

$$T_{aw} = T_c\,\frac{1 + r\,\frac{\gamma-1}{2}Ma^2}{1 + \frac{\gamma-1}{2}Ma^2}, \quad r = Pr^{1/3}$$

*Notes:* Gas viscosity via Sutherland's law. $h_g$ clamped to [100, 50 000] W m⁻² K⁻¹.
Fully-turbulent BL assumed from station 0.
""")

        with t[1]:
            st.markdown(r"""
### Gnielinski — Single-Phase Turbulent

$$Nu = \frac{(f/8)(Re-1000)\,Pr}{1 + 12.7\sqrt{f/8}\,(Pr^{2/3}-1)}, \quad
f = (0.79\ln Re - 1.64)^{-2}$$

Valid: $3\,000 < Re < 5\times10^6$, $0.5 < Pr < 2000$.  
Laminar ($Re<2300$): $Nu=3.66$. Transition: linear interpolation.

### Dittus-Boelter (fallback / non-N2O fluids)

$$Nu = 0.023\,Re^{0.8}\,Pr^{0.4}$$
""")

        with t[2]:
            st.markdown(r"""
### ONB Criterion — Bergles-Rohsenow

$$\Delta T_{\text{ONB}} = 0.556\left(\frac{q''}{1082}\right)^{0.463}
  \left(\frac{P}{10^6}\right)^{-0.535}$$

Boiling activates when $T_{w,\text{est}} - T_{\text{sat}} > \Delta T_{\text{ONB}}$.

### Subcooled Boiling Enhancement

$$h_{\text{sub}} = h_{\text{sp}}\!\left[1 + 0.5\!\left(\frac{\Delta T_w}{\Delta T_{\text{sub}}}\right)^{\!2}\right]$$

(capped at 5×)

### Saturated Boiling — Chen (1966)

$$h_{\text{tp}} = S\,h_{\text{nb}} + F\,h_{\text{lo}}$$

Lockhart-Martinelli parameter: $X_{tt} = \left(\frac{1-x}{x}\right)^{0.9}\!\left(\frac{\rho_v}{\rho_l}\right)^{0.5}\!\left(\frac{\mu_l}{\mu_v}\right)^{0.1}$

Enhancement: $F = 2.35\left(\tfrac{1}{X_{tt}}+0.213\right)^{0.736}$

Suppression: $S = \dfrac{1}{1 + 2.53\times10^{-6}\,Re_{\text{tp}}^{1.17}}$

Nucleate term (Forster-Zuber):
$h_{\text{nb}} \propto k_l^{0.79}\,c_{p,l}^{0.45}\,\rho_l^{0.49} / (\sigma^{0.5}\,\mu_l^{0.29}\,h_{fg}^{0.24}\,\rho_v^{0.24})$
""")

        with t[3]:
            st.markdown(r"""
### Critical Heat Flux — Bowring Correlation

$$q''_{\text{CHF}} = \frac{A\,(1-x)}{1 + C\,G\,D\,x / A}$$

$$A = \frac{2.317\,h_{fg}\,G^{0.5}}{1 + 0.0143\,D^{0.5}\,G}, \quad
C = 0.077\exp(-0.02\,P/10^5)$$

**Design rule enforced in RESA:**

$$\boxed{\frac{q''_{\text{actual}}}{q''_{\text{CHF}}} < 0.5}$$

Above $P_c$: no CHF ($q''_{\text{CHF}} = \infty$).

**Post-CHF:** film boiling, $h \approx 500$ W m⁻² K⁻¹ — wall burnout imminent.
""")

        with t[4]:
            st.markdown(r"""
### Jackson Correlation — Supercritical HT

$$Nu = 0.0183\,Re^{0.82}\,Pr_b^{0.5}\,\left(\frac{\rho_w}{\rho_b}\right)^{0.3}$$

Density ratio corrects for steep property gradients near $T_{pc}$.

### Heat Transfer Deterioration (HTD) Warning

| Trigger | Threshold |
|---------|-----------|
| Wall crosses $T_{pc}$ while bulk liquid-like | $T_b < T_{pc} < T_w$ |
| High heat-to-mass flux | $q''/G > 500$ W kg⁻¹ m |
| Near pseudo-critical | $|T_b - T_{pc}| < 10$ K |

When HTD risk detected → HTC conservatively halved.
""")

        with t[5]:
            st.markdown(r"""
### Friction — Colebrook-White (Haaland approximation)

$$\frac{1}{\sqrt{f}} = -1.8\log_{10}\!\left[\left(\frac{\varepsilon/D}{3.7}\right)^{1.11} + \frac{6.9}{Re}\right]$$

### Two-Phase Multiplier — Lockhart-Martinelli / Chisholm

$$\Delta P_{\text{tp}} = \phi_{lo}^2\,\Delta P_{lo}, \quad
\phi_{lo}^2 = 1 + \frac{C}{X_{tt}} + \frac{1}{X_{tt}^2} \quad (C=20,\text{ turb-turb})$$

### Acceleration Pressure Drop

$$\Delta P_{\text{acc}} = G^2\!\left(\frac{1}{\rho_{\text{out}}} - \frac{1}{\rho_{\text{in}}}\right)$$

Zivi void fraction: $\alpha = 1\big/\!\left[1 + \frac{1-x}{x}\frac{\rho_v}{\rho_l}S\right]$, $S=(\rho_l/\rho_v)^{1/3}$
""")

        with t[6]:
            st.markdown(r"""
### N2O Thermodynamic Properties

| Property | Source |
|----------|--------|
| $\rho$, $h$, $c_p$, saturation | CoolProp (REFPROP-quality EOS) |
| $\mu$ | Chapman-Enskog + density residual |
| $k$ | Modified Eucken + density correction |
| $\sigma$ | Guggenheim-Katayama |

**Critical constants:**

| | Value |
|-|-------|
| $T_c$ | 309.52 K (36.37 °C) |
| $P_c$ | 7.245 MPa (72.45 bar) |
| $\rho_c$ | 452 kg m⁻³ |
| $M$ | 44.013 g mol⁻¹ |

**Viscosity (Chapman-Enskog):** $\mu = \mu_0(T) + \mu_r(T,\rho)$, LJ params $\sigma_{LJ}=3.828$ Å, $\varepsilon/k=232.4$ K.

**Thermal conductivity (Eucken):** $k = \mu(c_v + f_{\text{int}}R) + k_r$, $f_{\text{int}}=1.32$.
""")

        with t[7]:
            st.markdown(r"""
### Key Modelling Assumptions

**Geometry & flow:**
- 1-D marching — axial only; radial / azimuthal gradients ignored
- Uniform circumferential heat flux at each station
- Rectangular channels: $D_h = 4wh\,/\,2(w+h)$
- Counter-flow default: coolant enters at nozzle exit

**Heat transfer:**
- Fully turbulent BL from station 0 (Bartz)
- 1-D wall conduction, constant $k_w$ (no T-dependence)
- Gnielinski valid to $Re = 2300$; laminar $Nu=3.66$ below
- Bartz $\sigma$ computed with $T_w/T_c = 0.6$ for stability

**CHF safety:**
- Bowring valid for uniform flux, horizontal/upward flow
- 2× safety factor ($q/q_{CHF} < 50\%$) is a **hard design constraint**
- Add 10–20 % margin to $\Delta P$ for minor losses (bends, manifolds)

**Supercritical:**
- $T_{pc}$ found by linear search over $[T_c, T_c+50\text{ K}]$, 50 pts
- HTD reduction of 0.5 is empirical — CFD validation recommended near $T_{pc}$
- No buoyancy effects (valid for high-$G$ forced convection)

**Material defaults:**
- Copper alloy: $k_w = 350$ W m⁻¹ K⁻¹, limit 800 K (527 °C)
- Inconel 718: $k_w \approx 15$ W m⁻¹ K⁻¹, limit 1100 K
""")


# ── Synthetic geometry helpers ─────────────────────────────────────────────

def _make_contour(r_ch, r_th, r_ex, L_ch, L_noz):
    x_th = L_ch
    def contour(x):
        if x <= x_th:
            t = x / x_th
            return r_ch - (r_ch - r_th) * (3*t**2 - 2*t**3)
        else:
            t = (x - x_th) / L_noz
            return r_th + (r_ex - r_th) * t
    return contour, x_th


def _make_q_profile(q_max, x_th, x_end, n=200):
    x = np.linspace(0, x_end, n)
    sc = x_th * 0.4
    sn = (x_end - x_th) * 0.5
    q = np.where(x <= x_th,
                 q_max * np.exp(-0.5*((x-x_th)/sc)**2),
                 q_max * np.exp(-0.5*((x-x_th)/sn)**2))
    q = np.maximum(q, 0.15 * q_max)
    return lambda xv: float(np.interp(xv, x, q)), x, q


# ── Plot builders ──────────────────────────────────────────────────────────

def _plot_temperature(result: CoolingAnalysisResult):
    xs     = [s.x for s in result.stations]
    T_bulk = [s.fluid.T - 273.15 for s in result.stations]
    T_hot  = [s.T_wall_hot - 273.15 for s in result.stations]
    T_cold = [s.T_wall_cold - 273.15 for s in result.stations]

    T_sat_pts = []
    try:
        from CoolProp.CoolProp import PropsSI
        for s in result.stations:
            if not s.fluid.is_supercritical and s.fluid.P > 5e5:
                try:
                    T_sat_pts.append((s.x, PropsSI("T","P",s.fluid.P,"Q",0,"N2O")-273.15))
                except Exception:
                    pass
    except ImportError:
        pass

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=T_hot,  name="T_wall hot-side",
                             line=dict(color="#ff5722", width=2)))
    fig.add_trace(go.Scatter(x=xs, y=T_cold, name="T_wall cold-side",
                             line=dict(color="#ff9d3a", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=xs, y=T_bulk, name="T_coolant bulk",
                             line=dict(color="#4a9eff", width=2.5)))
    if T_sat_pts:
        xs2, ys2 = zip(*T_sat_pts)
        fig.add_trace(go.Scatter(x=list(xs2), y=list(ys2), name="T_sat",
                                 line=dict(color="#1abc9c", width=1.5, dash="dot")))
    # Wall material limit
    fig.add_hline(y=527, line=dict(color="#ff4d4d", width=1, dash="dot"),
                  annotation_text="800 K wall limit")

    fig.update_layout(title="Temperature Profile", xaxis_title="x [m]",
                      yaxis_title="T [°C]", hovermode="x unified")
    return _apply_theme(fig)


def _plot_ht(result: CoolingAnalysisResult):
    xs     = [s.x for s in result.stations]
    h_conv = [s.h_conv for s in result.stations]
    q_MW   = [s.q_flux/1e6 for s in result.stations]
    chf_pc = [s.chf_margin*100 for s in result.stations]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("HTC & Heat Flux","CHF Ratio"),
                        vertical_spacing=0.14)
    fig.add_trace(go.Scatter(x=xs, y=h_conv, name="h_conv [W/m²K]",
                             line=dict(color="#4a9eff", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=q_MW, name="q\" [MW/m²]",
                             line=dict(color="#ff6b6b", width=2),
                             yaxis="y3"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=chf_pc, name="q/q_CHF [%]",
                             line=dict(color="#f0c040", width=2.5),
                             fill="tozeroy",
                             fillcolor="rgba(240,192,64,0.1)"), row=2, col=1)
    fig.add_hline(y=50, row=2, col=1,
                  line=dict(color="#ff4d4d", width=1.5, dash="dash"),
                  annotation_text="50 % CHF limit")
    fig.update_yaxes(title_text="h_conv [W/m²K]", row=1, col=1)
    fig.update_yaxes(title_text="q/q_CHF [%]", row=2, col=1)
    fig.update_xaxes(title_text="x [m]", row=2, col=1)
    fig.update_layout(title="Heat Transfer & CHF Margin", hovermode="x unified")
    return _apply_theme(fig)


def _plot_pq(result: CoolingAnalysisResult):
    xs    = [s.x for s in result.stations]
    P_bar = [s.fluid.P/1e5 for s in result.stations]
    qual  = [s.fluid.quality*100 if s.fluid.quality is not None else 0
             for s in result.stations]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Coolant Pressure","Vapour Quality"),
                        vertical_spacing=0.14)
    fig.add_trace(go.Scatter(x=xs, y=P_bar, name="P [bar]",
                             line=dict(color="#c084fc", width=2)), row=1, col=1)
    fig.add_hline(y=N2OConstants.P_CRIT/1e5, row=1, col=1,
                  line=dict(color="#f472b6", width=1, dash="dot"),
                  annotation_text="P_crit 72.45 bar")
    fig.add_trace(go.Scatter(x=xs, y=qual, name="Quality x [%]",
                             line=dict(color="#2ecc71", width=2),
                             fill="tozeroy",
                             fillcolor="rgba(46,204,113,0.1)"), row=2, col=1)
    fig.add_hline(y=80, row=2, col=1,
                  line=dict(color="#ff9d3a", width=1, dash="dash"),
                  annotation_text="x = 80 %")
    fig.update_yaxes(title_text="P [bar]", row=1, col=1)
    fig.update_yaxes(title_text="x [%]", row=2, col=1)
    fig.update_xaxes(title_text="x [m]", row=2, col=1)
    fig.update_layout(title="Pressure & Vapour Quality", hovermode="x unified")
    return _apply_theme(fig)


def _plot_regime(result: CoolingAnalysisResult):
    xs = [s.x for s in result.stations]
    rv = [s.flow_regime.value for s in result.stations]
    seen = list(dict.fromkeys(rv))
    fig = go.Figure()
    for reg in seen:
        xr = [x for x, r in zip(xs, rv) if r == reg]
        colour = _REGIME_COLOR.get(reg, "#888")
        fig.add_trace(go.Scatter(x=xr, y=[1]*len(xr), mode="markers",
                                 marker=dict(color=colour, size=12, symbol="square"),
                                 name=_REGIME_LABEL.get(reg, reg)))
    fig.update_layout(title="Flow Regime Map",
                      xaxis_title="x [m]",
                      yaxis=dict(showticklabels=False, showgrid=False, range=[0.5, 1.5]),
                      height=220, hovermode="x")
    return _apply_theme(fig)


def _plot_q_preview(x_arr, q_arr, x_th):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_arr, y=q_arr/1e6, mode="lines",
                             line=dict(color="#ff6b6b", width=2),
                             fill="tozeroy", fillcolor="rgba(255,107,107,0.1)",
                             name="q\" [MW/m²]"))
    fig.add_vline(x=x_th, line=dict(color="#f0c040", width=1, dash="dash"),
                  annotation_text="throat")
    fig.update_layout(title="Heat Flux Profile (input)",
                      xaxis_title="x [m]", yaxis_title="q\" [MW/m²]", height=230)
    return _apply_theme(fig)


def _plot_parametric(label, sv, results):
    T_out  = [r.T_outlet - 273.15 for r in results]
    dP     = [r.dP_total/1e5 for r in results]
    T_wall = [r.max_wall_temp - 273.15 for r in results]
    chf    = [r.min_chf_margin*100 for r in results]

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Outlet Temp [°C]","Pressure Drop [bar]",
                                        "Max Wall Temp [°C]","Max CHF Ratio [%]"),
                        vertical_spacing=0.20, horizontal_spacing=0.14)
    kw = dict(mode="lines+markers", marker=dict(size=7))
    fig.add_trace(go.Scatter(x=sv, y=T_out, line=dict(color="#4a9eff"), **kw), row=1, col=1)
    fig.add_trace(go.Scatter(x=sv, y=dP,    line=dict(color="#c084fc"), **kw), row=1, col=2)
    fig.add_trace(go.Scatter(x=sv, y=T_wall,line=dict(color="#ff6b6b"), **kw), row=2, col=1)
    fig.add_trace(go.Scatter(x=sv, y=chf,   line=dict(color="#f0c040"), **kw), row=2, col=2)
    fig.add_hline(y=527, row=2, col=1, line=dict(color="#ff4d4d", dash="dash", width=1),
                  annotation_text="800 K limit")
    fig.add_hline(y=50,  row=2, col=2, line=dict(color="#ff4d4d", dash="dash", width=1),
                  annotation_text="50 % limit")
    for r, c in [(1,1),(1,2),(2,1),(2,2)]:
        fig.update_xaxes(title_text=label, row=r, col=c)
    fig.update_layout(title=f"Parametric Sweep — {label}", showlegend=False, height=520)
    return _apply_theme(fig)


# ── Solver runner ──────────────────────────────────────────────────────────

def _run(m_dot, P_in, T_in, n_ch, w_th, ar, t_wall, k_wall,
         contour_fn, q_fn, x0, x1, n_st=120, direction="counter",
         name="Engine"):
    if not N2O_OK:
        st.error("N2O module unavailable — install CoolProp.")
        return None

    w = w_th
    h = w_th * ar

    geom = CoolingChannelGeometry(
        width=lambda x: w,
        height=lambda x: h,
        rib_width=lambda x: max(1e-4, 2*np.pi*contour_fn(x)/n_ch - w),
        wall_thickness=t_wall,
        k_wall=k_wall,
    )
    solver = N2OCoolingSolver(
        contour=contour_fn, channel_geom=geom, q_flux_profile=q_fn,
        m_dot=m_dot, P_inlet=P_in, T_inlet=T_in,
        x_start=x0, x_end=x1, n_stations=n_st,
        n_channels=n_ch, flow_direction=direction, engine_name=name,
    )
    try:
        return solver.solve()
    except Exception as exc:
        st.error(f"Solver error: {exc}")
        return None


# ── Results display ────────────────────────────────────────────────────────

def _show_results(result: CoolingAnalysisResult):
    chf_ok  = result.min_chf_margin < 0.5
    wall_ok = result.max_wall_temp < 800
    ok      = chf_ok and wall_ok and not result.errors

    bc1, bc2, bc3 = st.columns([1, 1, 2])
    with bc1:
        _badge("✓ CHF SAFE" if chf_ok else "✗ CHF EXCEEDED",
               "safe" if chf_ok else "danger")
    with bc2:
        _badge("✓ WALL OK"  if wall_ok else "✗ WALL OVERTEMP",
               "safe" if wall_ok else "danger")
    with bc3:
        _badge("✓ DESIGN FEASIBLE" if ok else "⚠  REVIEW REQUIRED",
               "safe" if ok else "warn")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    dT = result.T_outlet - result.T_inlet
    with c1: _kpi("Outlet Temp",   f"{result.T_outlet-273.15:.1f} °C", f"+{dT:.1f} K rise")
    with c2: _kpi("Pressure Drop", f"{result.dP_total/1e5:.2f} bar",
                                   f"In: {result.P_inlet/1e5:.1f} bar")
    with c3: _kpi("Max CHF Ratio", f"{result.min_chf_margin*100:.1f} %", "limit: 50 %",
                  color="#21c97a" if chf_ok else "#d64045")
    with c4: _kpi("Max Wall Temp", f"{result.max_wall_temp-273.15:.0f} °C",
                  f"({result.max_wall_temp:.0f} K)",
                  color="#21c97a" if wall_ok else "#d64045")
    with c5: _kpi("Heat Absorbed", f"{result.Q_total/1e3:.1f} kW",
                  f"Max quality: {result.max_quality*100:.1f} %")

    st.markdown("<br>", unsafe_allow_html=True)

    if result.regimes_encountered:
        st.markdown("**Regimes encountered:**")
        cols = st.columns(min(len(result.regimes_encountered), 6))
        for col, reg in zip(cols, result.regimes_encountered):
            clr = _REGIME_COLOR.get(reg.value, "#888")
            col.markdown(
                f'<span style="background:{clr}22;color:{clr};border:1px solid {clr};'
                f'border-radius:5px;padding:2px 8px;font-size:0.73rem;">'
                f'{_REGIME_LABEL.get(reg.value, reg.value)}</span>',
                unsafe_allow_html=True)

    st.divider()
    _section("Results Plots")
    tab1, tab2, tab3, tab4 = st.tabs(["Temperature", "Heat Transfer",
                                       "Pressure & Quality", "Regime Map"])
    with tab1: st.plotly_chart(_plot_temperature(result), use_container_width=True)
    with tab2: st.plotly_chart(_plot_ht(result),          use_container_width=True)
    with tab3: st.plotly_chart(_plot_pq(result),          use_container_width=True)
    with tab4: st.plotly_chart(_plot_regime(result),      use_container_width=True)

    if result.warnings:
        with st.expander(f"{len(result.warnings)} Warnings"):
            for w in result.warnings[:50]:
                st.markdown(f"<small style='color:#f0a000;'>• {w}</small>",
                            unsafe_allow_html=True)
    for e in result.errors:
        st.error(e)

    with st.expander("Station Data Table"):
        import pandas as pd
        step = max(1, len(result.stations)//50)
        rows = [{"x [m]": f"{s.x:.4f}",
                 "T_bulk [°C]": f"{s.fluid.T-273.15:.1f}",
                 "T_wall_hot [°C]": f"{s.T_wall_hot-273.15:.1f}",
                 "P [bar]": f"{s.fluid.P/1e5:.2f}",
                 "Quality": f"{s.fluid.quality:.3f}" if s.fluid.quality else "—",
                 "h_conv [W/m²K]": f"{s.h_conv:.0f}",
                 "CHF %": f"{s.chf_margin*100:.1f}",
                 "Regime": _REGIME_LABEL.get(s.flow_regime.value, s.flow_regime.value)}
                for s in result.stations[::step]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=320)


# ── Main entry point ───────────────────────────────────────────────────────

def render_cooling_page():
    st.markdown(
        '<h1 style="font-size:1.9rem;">Regenerative Cooling Analysis</h1>'
        '<p style="color:#4a6a8a;margin-top:-0.4rem;">'
        'Two-phase N₂O cooling — full phase-aware heat transfer physics</p>',
        unsafe_allow_html=True,
    )

    if not N2O_OK:
        st.error("N2O cooling module could not be imported. "
                 "Ensure CoolProp is installed (`pip install CoolProp`).")
        return
    if not PLOTLY_OK:
        st.error("Plotly is required. Install with `pip install plotly`.")
        return

    _render_model_docs()
    st.divider()

    mode = st.radio("Analysis mode",
                    ["Quick Analysis", "Full Engine (from design)", "Parametric Sweep"],
                    horizontal=True)
    st.divider()

    # ══════════════════════════════════════════════════════════════════
    # QUICK ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    if mode == "Quick Analysis":
        _section("Operating Conditions")
        c1, c2, c3 = st.columns(3)
        with c1:
            P_in  = st.number_input("Inlet pressure [bar]", 10.0, 200.0, 60.0, 5.0)
            T_in  = st.number_input("Inlet temperature [°C]", -60.0, 25.0, -10.0, 5.0)
            m_dot = st.number_input("Mass flow [kg/s]", 0.05, 20.0, 0.5, 0.05)
        with c2:
            q_pk   = st.number_input("Peak heat flux [MW/m²]", 0.5, 100.0, 15.0, 0.5)
            dirn   = st.selectbox("Flow direction", ["counter", "co"])
            n_stat = st.slider("Solver stations", 50, 300, 120, 10)
        with c3:
            st.markdown("**Material (wall)**")
            mat = st.selectbox("Material", ["Copper alloy (350 W/mK)",
                                            "Inconel 718 (15 W/mK)",
                                            "Steel 316L (16 W/mK)",
                                            "Custom"])
            k_wall_map = {"Copper alloy (350 W/mK)": 350.0,
                          "Inconel 718 (15 W/mK)": 15.0,
                          "Steel 316L (16 W/mK)": 16.0, "Custom": None}
            kw = k_wall_map[mat]
            if kw is None:
                kw = st.number_input("Custom k_wall [W/mK]", 5.0, 500.0, 50.0, 5.0)

        _section("Engine Contour")
        c1, c2 = st.columns(2)
        with c1:
            r_ch = st.number_input("Chamber radius [m]", 0.01, 0.5, 0.06, 0.005)
            r_th = st.number_input("Throat radius [m]",  0.003, 0.3, 0.028, 0.001)
            r_ex = st.number_input("Exit radius [m]",    0.01, 1.0, 0.09, 0.005)
        with c2:
            L_ch  = st.number_input("Chamber length [m]", 0.05, 1.0, 0.12, 0.01)
            L_noz = st.number_input("Nozzle length [m]",  0.03, 0.5, 0.12, 0.01)

        _section("Channel Geometry")
        c1, c2 = st.columns(2)
        with c1:
            n_ch    = int(st.number_input("Number of channels", 10, 200, 40, 5, format="%d"))
            w_mm    = st.number_input("Channel width [mm]", 0.2, 5.0, 1.0, 0.1)
            ar      = st.number_input("Aspect ratio h/w", 1.0, 10.0, 2.5, 0.5)
        with c2:
            t_mm    = st.number_input("Wall thickness [mm]", 0.2, 5.0, 0.8, 0.1)

        x_end = L_ch + L_noz
        cfn, x_th = _make_contour(r_ch, r_th, r_ex, L_ch, L_noz)
        qfn, xa, qa = _make_q_profile(q_pk*1e6, x_th, x_end)

        with st.expander("Preview heat flux profile"):
            st.plotly_chart(_plot_q_preview(xa, qa, x_th), use_container_width=True)

        if st.button("▶  Run Cooling Analysis", type="primary"):
            with st.spinner("Running 1-D marching solver…"):
                res = _run(m_dot, P_in*1e5, T_in+273.15,
                           n_ch, w_mm*1e-3, ar, t_mm*1e-3, kw,
                           cfn, qfn, 0.0, x_end, n_stat, dirn)
            if res:
                st.session_state["_cooling_result_qa"] = res
                _show_results(res)
        elif "_cooling_result_qa" in st.session_state:
            _show_results(st.session_state["_cooling_result_qa"])

    # ══════════════════════════════════════════════════════════════════
    # FULL ENGINE
    # ══════════════════════════════════════════════════════════════════
    elif mode == "Full Engine (from design)":
        dr  = st.session_state.get("design_result")
        cfg = st.session_state.get("engine_config")
        if dr is None or cfg is None:
            st.warning("No engine design in session. Run **Engine Design** first.")
            if st.button("→ Go to Engine Design"):
                st.session_state.current_page = "design"
                st.rerun()
            return

        st.info(f"Engine: **{cfg.engine_name}** — "
                f"Thrust {cfg.thrust_n:.0f} N, Pc {cfg.pc_bar:.1f} bar, "
                f"ṁ = {dr.massflow_total:.3f} kg/s")

        _section("Cooling Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            P_in  = st.number_input("Inlet pressure [bar]",
                                    float(cfg.pc_bar)*1.2, float(cfg.pc_bar)*3.0,
                                    float(getattr(cfg,"coolant_p_in_bar", cfg.pc_bar*1.5)), 5.0)
            T_in  = st.number_input("Inlet temperature [°C]", -60.0, 25.0, -10.0, 5.0)
        with c2:
            mfrac = st.slider("Coolant mass fraction", 0.05, 1.0,
                              float(getattr(cfg,"cooling_mass_fraction",0.3)), 0.05)
            n_ch  = int(st.number_input("Channels", 10, 200, 40, 5, format="%d"))
        with c3:
            w_mm  = st.number_input("Channel width [mm]", 0.2, 5.0, 1.0, 0.1)
            ar    = st.number_input("Aspect ratio", 1.0, 8.0, 2.5, 0.5)
            t_mm  = st.number_input("Wall thickness [mm]", 0.2, 5.0, 0.8, 0.1)

        # Build contour from nozzle geometry
        ng = getattr(dr, "nozzle_geometry", None)
        if ng is not None and hasattr(ng, "x"):
            xn, rn = np.array(ng.x), np.array(ng.r)
            cfn = lambda x, _x=xn, _r=rn: float(np.interp(x, _x, _r))
            x0, x1 = float(xn[0]), float(xn[-1])
            x_th   = float(xn[np.argmin(rn)])
        else:
            r_th = dr.dt_mm/2000; r_ex = dr.de_mm/2000
            Ln   = dr.length_mm/1000; Lc = Ln*0.4
            cfn, x_th = _make_contour(r_th*2, r_th, r_ex, Lc, Ln)
            x0, x1 = 0.0, Lc+Ln

        # Heat flux: from design or synthetic
        hfp = getattr(dr, "heat_flux_profile", None)
        if hfp is not None and hasattr(hfp, "x"):
            xh, qh = np.array(hfp.x), np.array(hfp.q)
            qfn = lambda x, _x=xh, _q=qh: float(np.interp(x, _x, _q))
        else:
            qfn, _, _ = _make_q_profile(15e6, x_th, x1)

        m_dot = dr.massflow_total * mfrac

        if st.button("▶  Run Full-Engine Cooling", type="primary"):
            with st.spinner("Solving…"):
                res = _run(m_dot, P_in*1e5, T_in+273.15,
                           n_ch, w_mm*1e-3, ar, t_mm*1e-3, 350.0,
                           cfn, qfn, x0, x1, name=cfg.engine_name)
            if res:
                _show_results(res)

    # ══════════════════════════════════════════════════════════════════
    # PARAMETRIC SWEEP
    # ══════════════════════════════════════════════════════════════════
    elif mode == "Parametric Sweep":
        _section("Baseline Configuration")
        c1, c2, c3 = st.columns(3)
        with c1:
            P0    = st.number_input("Baseline inlet pressure [bar]", 20.0, 200.0, 60.0, 5.0)
            T0    = st.number_input("Inlet temperature [°C]", -60.0, 25.0, -10.0, 5.0)
            m0    = st.number_input("Baseline mass flow [kg/s]", 0.1, 20.0, 0.5, 0.05)
        with c2:
            r_ch  = st.number_input("Chamber radius [m]", 0.01, 0.5, 0.06, 0.005)
            r_th  = st.number_input("Throat radius [m]", 0.003, 0.3, 0.028, 0.001)
            L_tot = st.number_input("Total length [m]", 0.05, 1.0, 0.24, 0.01)
        with c3:
            n0    = int(st.number_input("Channels (baseline)", 10, 200, 40, 5, format="%d"))
            w0    = st.number_input("Throat width [mm]", 0.2, 5.0, 1.0, 0.1)
            q_pk  = st.number_input("Peak heat flux [MW/m²]", 0.5, 100.0, 15.0, 0.5)

        _section("Sweep Parameter")
        c1, c2 = st.columns(2)
        with c1:
            sweep = st.selectbox("Sweep parameter",
                                 ["Inlet Pressure [bar]","Mass Flow [kg/s]",
                                  "Channel Count","Throat Width [mm]","Aspect Ratio"])
            n_pts = st.slider("Number of points", 4, 20, 8)
        with c2:
            if   sweep == "Inlet Pressure [bar]": lo,hi = st.slider("Range [bar]",  20,200,(30,120))
            elif sweep == "Mass Flow [kg/s]":      lo,hi = st.slider("Range [kg/s]", 0.1,10.0,(0.2,2.0))
            elif sweep == "Channel Count":          lo,hi = st.slider("Range",        10,150,(20,80))
            elif sweep == "Throat Width [mm]":      lo,hi = st.slider("Range [mm]",  0.3,5.0,(0.5,3.0))
            else:                                   lo,hi = st.slider("AR range",    1.0,8.0,(1.5,5.0))

        if st.button("▶  Run Parametric Sweep", type="primary"):
            Lc = L_tot*0.4; Ln = L_tot*0.6; r_ex = r_th*3.5
            cfn, x_th = _make_contour(r_ch, r_th, r_ex, Lc, Ln)
            qfn, _, _ = _make_q_profile(q_pk*1e6, x_th, L_tot)
            sv = np.linspace(lo, hi, n_pts)
            results, ok_sv = [], []

            prog = st.progress(0.0, text="Running…")
            for i, v in enumerate(sv):
                prog.progress((i+1)/n_pts, text=f"Point {i+1}/{n_pts}")
                Pi  = v*1e5   if sweep=="Inlet Pressure [bar]" else P0*1e5
                mi  = v       if sweep=="Mass Flow [kg/s]"     else m0
                nci = int(v)  if sweep=="Channel Count"        else n0
                wi  = v*1e-3  if sweep=="Throat Width [mm]"    else w0*1e-3
                ari = v       if sweep=="Aspect Ratio"          else 2.5

                r = _run(mi, Pi, T0+273.15, nci, wi, ari, 0.8e-3, 350.0,
                         cfn, qfn, 0.0, L_tot, 80)
                if r:
                    results.append(r); ok_sv.append(float(v))

            prog.empty()
            if results:
                st.plotly_chart(_plot_parametric(sweep, ok_sv, results),
                                use_container_width=True)
            else:
                st.error("All sweep points failed. Check inputs.")
