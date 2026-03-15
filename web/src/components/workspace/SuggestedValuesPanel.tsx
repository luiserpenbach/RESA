import { useEffect, useRef } from "react";
import type { EngineDesignResponse, EngineConfigRequest } from "../../types/engine";

// Suppress Plotly type issues — loaded as UMD global
declare const Plotly: {
  newPlot: (el: HTMLElement, data: unknown[], layout: unknown, config?: unknown) => void;
  purge: (el: HTMLElement) => void;
};

interface SuggestedValuesPanelProps {
  config: EngineConfigRequest;
  result: EngineDesignResponse | null;
}

// ---------------------------------------------------------------------------
// Reference data (Sutton RPE 9th ed. Table 8-2 + Barrère, Jaumotte et al.)
// ---------------------------------------------------------------------------

const L_STAR_DATA: Record<string, { label: string; min: number; max: number; fuels: string[]; oxs: string[] }> = {
  "Ethanol / N2O":   { label: "Ethanol / N2O",   min: 900,  max: 1300, fuels: ["Ethanol90","Ethanol80","Ethanol","IPA"], oxs: ["N2O","Nitrous"] },
  "RP-1 / LOX":      { label: "RP-1 / LOX",       min: 1000, max: 1500, fuels: ["RP1","Kerosene"],                        oxs: ["LOX","O2"] },
  "Methane / LOX":   { label: "Methane / LOX",     min: 800,  max: 1200, fuels: ["Methane","LCH4"],                        oxs: ["LOX","O2"] },
  "LH2 / LOX":       { label: "LH2 / LOX",         min: 700,  max: 1100, fuels: ["LH2"],                                   oxs: ["LOX","O2"] },
  "Ethanol / H2O2":  { label: "Ethanol / H2O2",   min: 1000, max: 1400, fuels: ["Ethanol90","Ethanol80","Ethanol"],       oxs: ["H2O2"] },
  "Propane / LOX":   { label: "Propane / LOX",     min: 750,  max: 1150, fuels: ["Propane"],                               oxs: ["LOX","O2"] },
};

function getCurrentLstarKey(fuel: string, oxidizer: string): string | null {
  for (const [key, entry] of Object.entries(L_STAR_DATA)) {
    if (entry.fuels.includes(fuel) && entry.oxs.includes(oxidizer)) return key;
  }
  return null;
}

// Contraction ratio guidance (Huzel & Huang, Design of Liquid Propellant Rocket Engines)
const CR_DT_MM = [5, 10, 15, 20, 30, 40, 60, 80, 120, 200, 350];
const CR_MIN   = [4.0, 3.5, 3.2, 3.0, 2.8, 2.7, 2.6, 2.5, 2.5, 2.5, 2.5];
const CR_TYP   = [9.0, 7.0, 6.0, 5.5, 4.8, 4.3, 3.8, 3.5, 3.2, 3.0, 2.8];
const CR_MAX   = [16,  13,  11,  9.5, 8.0, 7.0, 6.0, 5.5, 5.0, 4.5, 4.0];

// ---------------------------------------------------------------------------
// Dark-theme Plotly layout defaults
// ---------------------------------------------------------------------------

const DARK_LAYOUT = {
  paper_bgcolor: "#0d1117",
  plot_bgcolor: "#0d2137",
  font: { color: "#8baac8", family: "monospace", size: 11 },
  xaxis: { gridcolor: "#1a3050", zerolinecolor: "#1a3050" },
  yaxis: { gridcolor: "#1a3050", zerolinecolor: "#1a3050" },
  margin: { l: 55, r: 20, t: 50, b: 55 },
};

const PLOTLY_CONFIG = { responsive: true, displaylogo: false, scrollZoom: false };

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function SuggestedValuesPanel({ config, result }: SuggestedValuesPanelProps) {
  const lstarRef = useRef<HTMLDivElement>(null);
  const crRef = useRef<HTMLDivElement>(null);

  const currentKey = getCurrentLstarKey(config.fuel, config.oxidizer);
  const dt = result?.dt_mm ?? null;

  useEffect(() => {
    if (!lstarRef.current) return;

    const entries = Object.values(L_STAR_DATA);
    const labels = entries.map((e) => e.label);
    const centers = entries.map((e) => (e.min + e.max) / 2);
    const errors = entries.map((e) => (e.max - e.min) / 2);
    const colors = entries.map((e) =>
      currentKey && e.label === L_STAR_DATA[currentKey]?.label
        ? "#4a9eff"
        : "rgba(100,140,180,0.55)"
    );

    const data: unknown[] = [
      {
        type: "bar",
        orientation: "h",
        y: labels,
        x: centers,
        error_x: {
          type: "data",
          array: errors,
          visible: true,
          color: "rgba(140,180,220,0.6)",
          thickness: 2,
          width: 6,
        },
        marker: { color: colors, line: { color: "transparent" } },
        hovertemplate: "%{y}<br>L* range: %{customdata[0]}–%{customdata[1]} mm<extra></extra>",
        customdata: entries.map((e) => [e.min, e.max]),
      },
    ];

    // Design L* marker line
    const designLstar = config.L_star;
    const shapes: unknown[] = [
      {
        type: "line",
        x0: designLstar, x1: designLstar,
        y0: -0.5, y1: entries.length - 0.5,
        line: { color: "#ffd740", width: 2, dash: "dash" },
      },
    ];

    const annotations: unknown[] = [
      {
        x: designLstar,
        y: entries.length - 0.5,
        text: `Design L* = ${designLstar.toFixed(0)} mm`,
        showarrow: false,
        font: { color: "#ffd740", size: 10 },
        xanchor: "left",
        yanchor: "top",
        xshift: 6,
      },
    ];

    const layout = {
      ...DARK_LAYOUT,
      title: { text: "L* Reference Values  (Sutton RPE 9th ed., Table 8-2)", font: { size: 12, color: "#8baac8" }, x: 0.5 },
      xaxis: { ...DARK_LAYOUT.xaxis, title: "Characteristic Chamber Length L* [mm]", range: [400, 1800] },
      yaxis: { ...DARK_LAYOUT.yaxis, automargin: true },
      shapes,
      annotations,
      height: 280,
      margin: { l: 110, r: 20, t: 50, b: 55 },
    };

    Plotly.newPlot(lstarRef.current, data, layout, PLOTLY_CONFIG);
    return () => { if (lstarRef.current) Plotly.purge(lstarRef.current); };
  }, [config.L_star, config.fuel, config.oxidizer, currentKey]);

  useEffect(() => {
    if (!crRef.current) return;

    const data: unknown[] = [
      {
        type: "scatter", mode: "lines",
        x: CR_DT_MM, y: CR_MAX,
        name: "Maximum CR",
        line: { color: "rgba(100,140,180,0.35)", width: 1, dash: "dot" },
        fill: "none",
        hovertemplate: "Dt=%{x} mm<br>CR_max=%{y:.1f}<extra></extra>",
      },
      {
        type: "scatter", mode: "lines",
        x: CR_DT_MM, y: CR_TYP,
        name: "Typical CR",
        line: { color: "#4a9eff", width: 2 },
        fill: "tonexty",
        fillcolor: "rgba(74,158,255,0.06)",
        hovertemplate: "Dt=%{x} mm<br>CR_typ=%{y:.1f}<extra></extra>",
      },
      {
        type: "scatter", mode: "lines",
        x: CR_DT_MM, y: CR_MIN,
        name: "Minimum CR",
        line: { color: "rgba(100,140,180,0.35)", width: 1, dash: "dot" },
        fill: "tonexty",
        fillcolor: "rgba(74,158,255,0.04)",
        hovertemplate: "Dt=%{x} mm<br>CR_min=%{y:.1f}<extra></extra>",
      },
    ];

    // Design point marker
    if (dt !== null && config.contraction_ratio) {
      data.push({
        type: "scatter", mode: "markers",
        x: [dt], y: [config.contraction_ratio],
        name: "Design point",
        marker: { size: 12, color: "#ffd740", symbol: "star", line: { color: "#000", width: 1 } },
        hovertemplate: `Design: Dt=${dt.toFixed(1)} mm, CR=${config.contraction_ratio}<extra></extra>`,
      } as unknown);
    }

    const layout = {
      ...DARK_LAYOUT,
      title: { text: "Contraction Ratio Guidelines  (Huzel & Huang)", font: { size: 12, color: "#8baac8" }, x: 0.5 },
      xaxis: { ...DARK_LAYOUT.xaxis, title: "Throat Diameter Dt [mm]", type: "log" as const },
      yaxis: { ...DARK_LAYOUT.yaxis, title: "Contraction Ratio Ac/At" },
      showlegend: true,
      legend: { x: 0.98, y: 0.98, xanchor: "right", bgcolor: "transparent", font: { size: 10 } },
      height: 280,
    };

    Plotly.newPlot(crRef.current, data, layout, PLOTLY_CONFIG);
    return () => { if (crRef.current) Plotly.purge(crRef.current); };
  }, [config.contraction_ratio, dt]);

  return (
    <div style={{ padding: 16, overflowY: "auto", height: "100%" }}>
      {/* Context note */}
      {currentKey && (
        <div style={{
          marginBottom: 12,
          padding: "8px 12px",
          background: "rgba(74,158,255,0.08)",
          border: "1px solid rgba(74,158,255,0.2)",
          borderRadius: 4,
          fontSize: 10,
          color: "var(--text-secondary)",
          fontFamily: "var(--font-mono)",
        }}>
          Current propellant combination: <strong style={{ color: "#4a9eff" }}>{currentKey}</strong>
          {" · "}Design L* = <strong style={{ color: "#ffd740" }}>{config.L_star.toFixed(0)} mm</strong>
          {dt !== null && ` · Dt = ${dt.toFixed(1)} mm`}
          {" · "}CR = <strong>{config.contraction_ratio}</strong>
        </div>
      )}

      {/* L* Chart */}
      <div style={{
        background: "var(--bg-elevated)",
        border: "1px solid var(--border)",
        borderRadius: 4,
        marginBottom: 12,
        overflow: "hidden",
      }}>
        <div ref={lstarRef} style={{ width: "100%" }} />
      </div>

      {/* CR Chart */}
      <div style={{
        background: "var(--bg-elevated)",
        border: "1px solid var(--border)",
        borderRadius: 4,
        marginBottom: 12,
        overflow: "hidden",
      }}>
        <div ref={crRef} style={{ width: "100%" }} />
      </div>

      {/* Source notes */}
      <div style={{
        fontSize: 9,
        color: "var(--text-muted)",
        fontFamily: "var(--font-mono)",
        lineHeight: 1.6,
      }}>
        Sources: G.P. Sutton, "Rocket Propulsion Elements" 9th ed. (Table 8-2);
        D.K. Huzel & D.H. Huang, "Design of Liquid Propellant Rocket Engines" 2nd ed.;
        Barrère, Jaumotte et al., "Rocket Propulsion".
        Values are typical ranges for steady-state liquid bipropellant engines.
        Monopropellant and solid designs may differ significantly.
      </div>
    </div>
  );
}
