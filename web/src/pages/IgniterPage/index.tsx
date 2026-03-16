import { useState, useCallback } from "react";
import { Icon } from "@blueprintjs/core";
import { useIgniterMutation } from "../../api/igniter";
import type { IgniterConfig, IgniterResponse } from "../../types/igniter";
import { DEFAULT_IGNITER_CONFIG } from "../../types/igniter";

function extractError(err: unknown): string {
  if (err && typeof err === "object" && "response" in err) {
    const resp = (err as { response?: { data?: { detail?: unknown } } }).response;
    if (typeof resp?.data?.detail === "string") return resp.data.detail;
  }
  if (err instanceof Error) return err.message;
  return "Igniter design failed.";
}

const thStyle: React.CSSProperties = {
  textAlign: "left", padding: "4px 8px",
  borderBottom: "1px solid var(--border-subtle)",
  color: "var(--text-muted)", fontWeight: 600,
};
const tdStyle: React.CSSProperties = {
  padding: "3px 8px", fontFamily: "var(--font-mono)",
  borderBottom: "1px solid var(--border-subtle)",
};

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
      <span style={{ color: "var(--text-muted)" }}>{label}</span>
      <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>{value}</span>
    </div>
  );
}

function SectionLabel({ text }: { text: string }) {
  return (
    <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginTop: 12, marginBottom: 4 }}>
      {text}
    </div>
  );
}

export default function IgniterPage() {
  const [config, setConfig] = useState<IgniterConfig>({ ...DEFAULT_IGNITER_CONFIG });
  const [result, setResult] = useState<IgniterResponse | null>(null);
  const [activeTab, setActiveTab] = useState("combustion");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const mutation = useIgniterMutation();
  const isRunning = mutation.isPending;

  const handleRun = useCallback(async () => {
    setErrorMsg(null);
    try {
      const res = await mutation.mutateAsync(config);
      setResult(res);
    } catch (err) {
      setErrorMsg(extractError(err));
    }
  }, [config, mutation]);

  const set = (p: Partial<IgniterConfig>) => setConfig((c) => ({ ...c, ...p }));

  return (
    <>
      <header className="app-topbar">
        <div className="topbar-logo"><div className="topbar-logo-dot" />RESA</div>
        <div className="topbar-breadcrumb"><span className="page-name">Igniter Design</span></div>
        <div className="topbar-spacer" />
        <div className="topbar-actions">
          <button className={`run-btn ${isRunning ? "is-running" : ""}`} onClick={handleRun} disabled={isRunning}>
            <Icon icon={isRunning ? "dot" : "play"} size={12} />
            {isRunning ? "COMPUTING..." : "DESIGN IGNITER"}
          </button>
        </div>
      </header>

      <div className="app-left-panel">
        <div style={{ flexShrink: 0, padding: "7px 12px", borderBottom: "1px solid var(--border-subtle)" }}>
          <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)" }}>
            Configuration
          </span>
        </div>
        <div className="panel-scroll" style={{ padding: "12px 16px" }}>
          <SectionLabel text="Combustion" />
          <div>
            <label className="form-label">Chamber Pressure [bar]</label>
            <input className="form-input" type="number" step="0.5" value={config.chamber_pressure_pa / 1e5}
              onChange={(e) => set({ chamber_pressure_pa: Number(e.target.value) * 1e5 })} />
          </div>
          <div>
            <label className="form-label">Mixture Ratio (O/F)</label>
            <input className="form-input" type="number" step="0.1" value={config.mixture_ratio}
              onChange={(e) => set({ mixture_ratio: Number(e.target.value) })} />
          </div>
          <div>
            <label className="form-label">Total Mass Flow [g/s]</label>
            <input className="form-input" type="number" step="1" value={config.total_mass_flow_kg_s * 1000}
              onChange={(e) => set({ total_mass_flow_kg_s: Number(e.target.value) / 1000 })} />
          </div>

          <SectionLabel text="Feed System" />
          <div>
            <label className="form-label">Ethanol Feed Pressure [bar]</label>
            <input className="form-input" type="number" step="0.5" value={config.ethanol_feed_pressure_pa / 1e5}
              onChange={(e) => set({ ethanol_feed_pressure_pa: Number(e.target.value) * 1e5 })} />
          </div>
          <div>
            <label className="form-label">N2O Feed Pressure [bar]</label>
            <input className="form-input" type="number" step="0.5" value={config.n2o_feed_pressure_pa / 1e5}
              onChange={(e) => set({ n2o_feed_pressure_pa: Number(e.target.value) * 1e5 })} />
          </div>
          <div>
            <label className="form-label">Ethanol Temp [K]</label>
            <input className="form-input" type="number" step="1" value={config.ethanol_feed_temperature_k}
              onChange={(e) => set({ ethanol_feed_temperature_k: Number(e.target.value) })} />
          </div>
          <div>
            <label className="form-label">N2O Temp [K]</label>
            <input className="form-input" type="number" step="1" value={config.n2o_feed_temperature_k}
              onChange={(e) => set({ n2o_feed_temperature_k: Number(e.target.value) })} />
          </div>

          <SectionLabel text="Chamber / Nozzle" />
          <div>
            <label className="form-label">L* [m]</label>
            <input className="form-input" type="number" step="0.1" value={config.l_star}
              onChange={(e) => set({ l_star: Number(e.target.value) })} />
          </div>
          <div>
            <label className="form-label">Expansion Ratio</label>
            <input className="form-input" type="number" step="0.5" value={config.expansion_ratio}
              onChange={(e) => set({ expansion_ratio: Number(e.target.value) })} />
          </div>

          <SectionLabel text="Injector" />
          <div>
            <label className="form-label">N2O Orifice Count</label>
            <input className="form-input" type="number" step="1" min="1" value={config.n2o_orifice_count}
              onChange={(e) => set({ n2o_orifice_count: Number(e.target.value) })} />
          </div>
          <div>
            <label className="form-label">Ethanol Orifice Count</label>
            <input className="form-input" type="number" step="1" min="1" value={config.ethanol_orifice_count}
              onChange={(e) => set({ ethanol_orifice_count: Number(e.target.value) })} />
          </div>
          <div>
            <label className="form-label">Discharge Coefficient</label>
            <input className="form-input" type="number" step="0.01" min="0.1" max="1" value={config.discharge_coefficient}
              onChange={(e) => set({ discharge_coefficient: Number(e.target.value) })} />
          </div>
        </div>
      </div>

      <div className="app-workspace">
        <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
          <div className="workspace-tabs">
            {["combustion", "geometry", "injector"].map((tab) => (
              <button key={tab} className={`workspace-tab ${activeTab === tab ? "active" : ""}`} onClick={() => setActiveTab(tab)}>
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
          <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
            {!result && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-muted)", fontSize: 12 }}>
                Configure parameters and design the igniter
              </div>
            )}
            {result && activeTab === "combustion" && (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead><tr>
                  <th style={thStyle}>Parameter</th><th style={thStyle}>Value</th>
                </tr></thead>
                <tbody>
                  <tr><td style={tdStyle}>Flame Temperature</td><td style={tdStyle}>{result.combustion.flame_temperature_k.toFixed(0)} K</td></tr>
                  <tr><td style={tdStyle}>C* (characteristic velocity)</td><td style={tdStyle}>{result.combustion.c_star_m_s.toFixed(1)} m/s</td></tr>
                  <tr><td style={tdStyle}>Gamma (γ)</td><td style={tdStyle}>{result.combustion.gamma.toFixed(4)}</td></tr>
                  <tr><td style={tdStyle}>Molecular Weight</td><td style={tdStyle}>{result.combustion.molecular_weight.toFixed(2)} kg/kmol</td></tr>
                  <tr><td style={tdStyle}>Thermal Power</td><td style={tdStyle}>{result.combustion.heat_power_kw.toFixed(2)} kW</td></tr>
                  <tr><td style={tdStyle}>Total Mass Flow</td><td style={tdStyle}>{(result.mass_flows.total_kg_s * 1000).toFixed(1)} g/s</td></tr>
                  <tr><td style={tdStyle}>Oxidizer (N2O)</td><td style={tdStyle}>{(result.mass_flows.oxidizer_kg_s * 1000).toFixed(1)} g/s</td></tr>
                  <tr><td style={tdStyle}>Fuel (Ethanol)</td><td style={tdStyle}>{(result.mass_flows.fuel_kg_s * 1000).toFixed(1)} g/s</td></tr>
                  <tr><td style={tdStyle}>Mixture Ratio (O/F)</td><td style={tdStyle}>{result.mass_flows.mixture_ratio.toFixed(3)}</td></tr>
                </tbody>
              </table>
            )}
            {result && activeTab === "geometry" && (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead><tr>
                  <th style={thStyle}>Dimension</th><th style={thStyle}>Value</th>
                </tr></thead>
                <tbody>
                  <tr><td style={tdStyle}>Chamber Diameter</td><td style={tdStyle}>{result.geometry.chamber_diameter_mm.toFixed(2)} mm</td></tr>
                  <tr><td style={tdStyle}>Chamber Length</td><td style={tdStyle}>{result.geometry.chamber_length_mm.toFixed(2)} mm</td></tr>
                  <tr><td style={tdStyle}>Chamber Volume</td><td style={tdStyle}>{result.geometry.chamber_volume_cm3.toFixed(2)} cm³</td></tr>
                  <tr><td style={tdStyle}>Throat Diameter</td><td style={tdStyle}>{result.geometry.throat_diameter_mm.toFixed(3)} mm</td></tr>
                  <tr><td style={tdStyle}>Exit Diameter</td><td style={tdStyle}>{result.geometry.exit_diameter_mm.toFixed(3)} mm</td></tr>
                  <tr><td style={tdStyle}>Nozzle Length</td><td style={tdStyle}>{result.geometry.nozzle_length_mm.toFixed(2)} mm</td></tr>
                </tbody>
              </table>
            )}
            {result && activeTab === "injector" && (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead><tr>
                  <th style={thStyle}>Parameter</th><th style={thStyle}>Value</th>
                </tr></thead>
                <tbody>
                  <tr><td style={tdStyle}>N2O Orifice Diameter</td><td style={tdStyle}>{result.injector.n2o_orifice_diameter_mm.toFixed(3)} mm</td></tr>
                  <tr><td style={tdStyle}>Ethanol Orifice Diameter</td><td style={tdStyle}>{result.injector.ethanol_orifice_diameter_mm.toFixed(3)} mm</td></tr>
                  <tr><td style={tdStyle}>N2O Injection Velocity</td><td style={tdStyle}>{result.injector.n2o_injection_velocity_m_s.toFixed(1)} m/s</td></tr>
                  <tr><td style={tdStyle}>Ethanol Injection Velocity</td><td style={tdStyle}>{result.injector.ethanol_injection_velocity_m_s.toFixed(1)} m/s</td></tr>
                  <tr><td style={tdStyle}>N2O Pressure Drop</td><td style={tdStyle}>{result.injector.n2o_pressure_drop_bar.toFixed(2)} bar</td></tr>
                  <tr><td style={tdStyle}>Ethanol Pressure Drop</td><td style={tdStyle}>{result.injector.ethanol_pressure_drop_bar.toFixed(2)} bar</td></tr>
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>

      <div className="app-right-panel">
        <div style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 12 }}>
            Key Results
          </div>
          {result ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <MetricRow label="Isp (theoretical)" value={`${result.performance.isp_theoretical_s.toFixed(1)} s`} />
              <MetricRow label="Thrust" value={`${result.performance.thrust_n.toFixed(2)} N`} />
              <MetricRow label="Flame Temp" value={`${result.combustion.flame_temperature_k.toFixed(0)} K`} />
              <MetricRow label="Thermal Power" value={`${result.combustion.heat_power_kw.toFixed(2)} kW`} />
              <MetricRow label="Chamber Ø" value={`${result.geometry.chamber_diameter_mm.toFixed(2)} mm`} />
              <MetricRow label="Throat Ø" value={`${result.geometry.throat_diameter_mm.toFixed(3)} mm`} />
              <MetricRow label="Mass Flow" value={`${(result.mass_flows.total_kg_s * 1000).toFixed(1)} g/s`} />
            </div>
          ) : (
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>No results yet</div>
          )}
        </div>
      </div>

      <footer className="app-statusbar">
        <div className="statusbar-item">
          <Icon icon={isRunning ? "dot" : errorMsg ? "cross" : result ? "tick-circle" : "circle"} size={10} />
          <span>{isRunning ? "COMPUTING" : errorMsg ? "ERROR" : result ? "NOMINAL" : "READY"}</span>
        </div>
        {errorMsg && <div className="statusbar-item" style={{ color: "var(--red)" }}>{errorMsg.slice(0, 80)}</div>}
        <div className="statusbar-spacer" />
        <div className="statusbar-item"><Icon icon="flame" size={10} /> RESA v2.0.0</div>
      </footer>
    </>
  );
}
