import { useState, useCallback } from "react";
import { Icon } from "@blueprintjs/core";
import { useInjectorMutation } from "../../api/injector";
import type { InjectorConfig, InjectorResponse } from "../../types/injector";
import { DEFAULT_INJECTOR_CONFIG } from "../../types/injector";

function extractError(err: unknown): string {
  if (err && typeof err === "object" && "response" in err) {
    const resp = (err as { response?: { data?: { detail?: unknown } } }).response;
    if (typeof resp?.data?.detail === "string") return resp.data.detail;
  }
  if (err instanceof Error) return err.message;
  return "Injector design failed.";
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

export default function InjectorPage() {
  const [config, setConfig] = useState<InjectorConfig>({ ...DEFAULT_INJECTOR_CONFIG, propellants: { ...DEFAULT_INJECTOR_CONFIG.propellants }, operating: { ...DEFAULT_INJECTOR_CONFIG.operating }, geometry: { ...DEFAULT_INJECTOR_CONFIG.geometry } });
  const [result, setResult] = useState<InjectorResponse | null>(null);
  const [activeTab, setActiveTab] = useState("geometry");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const mutation = useInjectorMutation();
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

  return (
    <>
      <header className="app-topbar">
        <div className="topbar-logo"><div className="topbar-logo-dot" />RESA</div>
        <div className="topbar-breadcrumb"><span className="page-name">Injector Design</span></div>
        <div className="topbar-spacer" />
        <div className="topbar-actions">
          <button className={`run-btn ${isRunning ? "is-running" : ""}`} onClick={handleRun} disabled={isRunning}>
            <Icon icon={isRunning ? "dot" : "play"} size={12} />
            {isRunning ? "COMPUTING..." : "DESIGN INJECTOR"}
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
          <SectionLabel text="Type" />
          <div>
            <label className="form-label">Injector Type</label>
            <select className="form-select" value={config.injector_type}
              onChange={(e) => setConfig((c) => ({ ...c, injector_type: e.target.value }))}>
              <option value="LCSC">LCSC (Liquid-Centered)</option>
              <option value="GCSC">GCSC (Gas-Centered)</option>
            </select>
          </div>

          <SectionLabel text="Operating Conditions" />
          <div>
            <label className="form-label">Inlet Pressure [bar]</label>
            <input className="form-input" type="number" step="1" value={config.operating.inlet_pressure / 1e5}
              onChange={(e) => setConfig((c) => ({ ...c, operating: { ...c.operating, inlet_pressure: Number(e.target.value) * 1e5 } }))} />
          </div>
          <div>
            <label className="form-label">Pressure Drop [bar]</label>
            <input className="form-input" type="number" step="1" value={config.operating.pressure_drop / 1e5}
              onChange={(e) => setConfig((c) => ({ ...c, operating: { ...c.operating, pressure_drop: Number(e.target.value) * 1e5 } }))} />
          </div>
          <div>
            <label className="form-label">Fuel Mass Flow [kg/s]</label>
            <input className="form-input" type="number" step="0.01" value={config.operating.mass_flow_fuel}
              onChange={(e) => setConfig((c) => ({ ...c, operating: { ...c.operating, mass_flow_fuel: Number(e.target.value) } }))} />
          </div>
          <div>
            <label className="form-label">Oxidizer Mass Flow [kg/s]</label>
            <input className="form-input" type="number" step="0.05" value={config.operating.mass_flow_oxidizer}
              onChange={(e) => setConfig((c) => ({ ...c, operating: { ...c.operating, mass_flow_oxidizer: Number(e.target.value) } }))} />
          </div>

          <SectionLabel text="Geometry" />
          <div>
            <label className="form-label">Number of Elements</label>
            <input className="form-input" type="number" step="1" min="1" value={config.geometry.num_elements}
              onChange={(e) => setConfig((c) => ({ ...c, geometry: { ...c.geometry, num_elements: Number(e.target.value) } }))} />
          </div>
          <div>
            <label className="form-label">Fuel Ports per Element</label>
            <input className="form-input" type="number" step="1" min="1" value={config.geometry.num_fuel_ports}
              onChange={(e) => setConfig((c) => ({ ...c, geometry: { ...c.geometry, num_fuel_ports: Number(e.target.value) } }))} />
          </div>
          <div>
            <label className="form-label">Spray Half-Angle [°]</label>
            <input className="form-input" type="number" step="5" min="10" max="80" value={config.geometry.spray_half_angle}
              onChange={(e) => setConfig((c) => ({ ...c, geometry: { ...c.geometry, spray_half_angle: Number(e.target.value) } }))} />
          </div>

          <SectionLabel text="Propellants" />
          <div>
            <label className="form-label">Fuel Temp [K]</label>
            <input className="form-input" type="number" step="5" value={config.propellants.fuel_temperature}
              onChange={(e) => setConfig((c) => ({ ...c, propellants: { ...c.propellants, fuel_temperature: Number(e.target.value) } }))} />
          </div>
          <div>
            <label className="form-label">Oxidizer Temp [K]</label>
            <input className="form-input" type="number" step="5" value={config.propellants.oxidizer_temperature}
              onChange={(e) => setConfig((c) => ({ ...c, propellants: { ...c.propellants, oxidizer_temperature: Number(e.target.value) } }))} />
          </div>
        </div>
      </div>

      <div className="app-workspace">
        <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
          <div className="workspace-tabs">
            {["geometry", "performance", "mass flows"].map((tab) => (
              <button key={tab} className={`workspace-tab ${activeTab === tab ? "active" : ""}`} onClick={() => setActiveTab(tab)}>
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
          <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
            {!result && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-muted)", fontSize: 12 }}>
                Configure parameters and design the injector
              </div>
            )}
            {result && activeTab === "geometry" && (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead><tr><th style={thStyle}>Dimension</th><th style={thStyle}>Value</th></tr></thead>
                <tbody>
                  <tr><td style={tdStyle}>Fuel Orifice Radius</td><td style={tdStyle}>{result.geometry.fuel_orifice_radius_mm.toFixed(3)} mm</td></tr>
                  <tr><td style={tdStyle}>Fuel Port Radius</td><td style={tdStyle}>{result.geometry.fuel_port_radius_mm.toFixed(3)} mm</td></tr>
                  <tr><td style={tdStyle}>Swirl Chamber Radius</td><td style={tdStyle}>{result.geometry.swirl_chamber_radius_mm.toFixed(3)} mm</td></tr>
                  <tr><td style={tdStyle}>Oxidizer Outlet Radius</td><td style={tdStyle}>{result.geometry.ox_outlet_radius_mm.toFixed(3)} mm</td></tr>
                  <tr><td style={tdStyle}>Oxidizer Inlet Orifice Radius</td><td style={tdStyle}>{result.geometry.ox_inlet_orifice_radius_mm.toFixed(3)} mm</td></tr>
                  <tr><td style={tdStyle}>Recess Length</td><td style={tdStyle}>{result.geometry.recess_length_mm.toFixed(3)} mm</td></tr>
                </tbody>
              </table>
            )}
            {result && activeTab === "performance" && (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead><tr><th style={thStyle}>Metric</th><th style={thStyle}>Value</th></tr></thead>
                <tbody>
                  <tr><td style={tdStyle}>Spray Half-Angle</td><td style={tdStyle}>{result.performance.spray_half_angle_deg.toFixed(1)} °</td></tr>
                  <tr><td style={tdStyle}>Swirl Number</td><td style={tdStyle}>{result.performance.swirl_number.toFixed(4)}</td></tr>
                  <tr><td style={tdStyle}>Momentum Flux Ratio (J)</td><td style={tdStyle}>{result.performance.momentum_flux_ratio.toFixed(4)}</td></tr>
                  <tr><td style={tdStyle}>Velocity Ratio</td><td style={tdStyle}>{result.performance.velocity_ratio.toFixed(4)}</td></tr>
                  <tr><td style={tdStyle}>Weber Number</td><td style={tdStyle}>{result.performance.weber_number.toFixed(2)}</td></tr>
                  <tr><td style={tdStyle}>Discharge Coefficient</td><td style={tdStyle}>{result.performance.discharge_coefficient.toFixed(4)}</td></tr>
                </tbody>
              </table>
            )}
            {result && activeTab === "mass flows" && (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead><tr><th style={thStyle}>Flow</th><th style={thStyle}>Value</th></tr></thead>
                <tbody>
                  <tr><td style={tdStyle}>Fuel / Element</td><td style={tdStyle}>{(result.mass_flows.fuel_per_element_kg_s * 1000).toFixed(2)} g/s</td></tr>
                  <tr><td style={tdStyle}>Oxidizer / Element</td><td style={tdStyle}>{(result.mass_flows.oxidizer_per_element_kg_s * 1000).toFixed(2)} g/s</td></tr>
                  <tr><td style={tdStyle}>Total Fuel</td><td style={tdStyle}>{(result.mass_flows.total_fuel_kg_s * 1000).toFixed(2)} g/s</td></tr>
                  <tr><td style={tdStyle}>Total Oxidizer</td><td style={tdStyle}>{(result.mass_flows.total_oxidizer_kg_s * 1000).toFixed(2)} g/s</td></tr>
                  <tr><td style={tdStyle}>Mixture Ratio (O/F)</td><td style={tdStyle}>{result.mass_flows.mixture_ratio.toFixed(3)}</td></tr>
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
              <MetricRow label="Type" value={result.injector_type} />
              <MetricRow label="Discharge Coeff." value={result.performance.discharge_coefficient.toFixed(4)} />
              <MetricRow label="Spray Half-Angle" value={`${result.performance.spray_half_angle_deg.toFixed(1)} °`} />
              <MetricRow label="Swirl Number" value={result.performance.swirl_number.toFixed(4)} />
              <MetricRow label="Total Fuel" value={`${(result.mass_flows.total_fuel_kg_s * 1000).toFixed(2)} g/s`} />
              <MetricRow label="Total Oxidizer" value={`${(result.mass_flows.total_oxidizer_kg_s * 1000).toFixed(2)} g/s`} />
              <MetricRow label="O/F Ratio" value={result.mass_flows.mixture_ratio.toFixed(3)} />
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
