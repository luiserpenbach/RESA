import { useState, useCallback, useMemo } from "react";
import { Icon } from "@blueprintjs/core";
import { useTankMutation } from "../../api/tank";
import type { TankSimConfig, TankSimResponse } from "../../types/tank";
import { DEFAULT_TANK_CONFIG } from "../../types/tank";
import { PlotlyRenderer } from "../../components/plots/PlotlyRenderer";

function extractError(err: unknown): string {
  if (err && typeof err === "object" && "response" in err) {
    const resp = (err as { response?: { data?: { detail?: unknown } } }).response;
    if (typeof resp?.data?.detail === "string") return resp.data.detail;
  }
  if (err instanceof Error) return err.message;
  return "Tank simulation failed.";
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

const DARK_LAYOUT: object = {
  paper_bgcolor: "#0d1117",
  plot_bgcolor: "#0d1117",
  font: { color: "#c9d1d9", size: 11 },
  margin: { t: 24, r: 16, b: 48, l: 56 },
  xaxis: { gridcolor: "#21262d", zerolinecolor: "#21262d" },
  yaxis: { gridcolor: "#21262d", zerolinecolor: "#21262d" },
};

function makeFigureJson(
  x: number[],
  traces: { y: number[]; name: string; color: string }[],
  xTitle: string,
  yTitle: string,
): string {
  return JSON.stringify({
    data: traces.map((t) => ({
      type: "scatter",
      mode: "lines",
      x,
      y: t.y,
      name: t.name,
      line: { color: t.color, width: 2 },
    })),
    layout: {
      ...DARK_LAYOUT,
      xaxis: { ...(DARK_LAYOUT as Record<string, object>).xaxis, title: xTitle },
      yaxis: { ...(DARK_LAYOUT as Record<string, object>).yaxis, title: yTitle },
    },
  });
}

export default function TankPage() {
  const [config, setConfig] = useState<TankSimConfig>({
    ...DEFAULT_TANK_CONFIG,
    tank: { ...DEFAULT_TANK_CONFIG.tank },
    pressurant: { ...DEFAULT_TANK_CONFIG.pressurant },
    propellant: { ...DEFAULT_TANK_CONFIG.propellant },
  });
  const [result, setResult] = useState<TankSimResponse | null>(null);
  const [activeTab, setActiveTab] = useState("pressure");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const mutation = useTankMutation();
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

  const setTank = (p: Partial<typeof config.tank>) => setConfig((c) => ({ ...c, tank: { ...c.tank, ...p } }));
  const setPres = (p: Partial<typeof config.pressurant>) => setConfig((c) => ({ ...c, pressurant: { ...c.pressurant, ...p } }));
  const setProp = (p: Partial<typeof config.propellant>) => setConfig((c) => ({ ...c, propellant: { ...c.propellant, ...p } }));

  const pressureFigure = useMemo(() => {
    if (!result) return null;
    return makeFigureJson(
      result.time_s,
      [{ y: result.pressure_bar, name: "Pressure", color: "#58a6ff" }],
      "Time [s]",
      "Pressure [bar]",
    );
  }, [result]);

  const massFigure = useMemo(() => {
    if (!result) return null;
    return makeFigureJson(
      result.time_s,
      [{ y: result.liquid_mass_kg, name: "Liquid Mass", color: "#3fb950" }],
      "Time [s]",
      "Liquid Mass [kg]",
    );
  }, [result]);

  return (
    <>
      <header className="app-topbar">
        <div className="topbar-logo"><div className="topbar-logo-dot" />RESA</div>
        <div className="topbar-breadcrumb"><span className="page-name">Tank Simulation</span></div>
        <div className="topbar-spacer" />
        <div className="topbar-actions">
          <button className={`run-btn ${isRunning ? "is-running" : ""}`} onClick={handleRun} disabled={isRunning}>
            <Icon icon={isRunning ? "dot" : "play"} size={12} />
            {isRunning ? "SIMULATING..." : "RUN SIMULATION"}
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
          <SectionLabel text="Tank Type" />
          <div>
            <label className="form-label">Propellant Type</label>
            <select className="form-select" value={config.tank_type}
              onChange={(e) => setConfig((c) => ({ ...c, tank_type: e.target.value }))}>
              <option value="n2o">N2O (Self-Pressurizing)</option>
              <option value="ethanol">Ethanol (Pressurized)</option>
            </select>
          </div>
          <div>
            <label className="form-label">Simulation Duration [s]</label>
            <input className="form-input" type="number" step="5" min="1" value={config.duration_s}
              onChange={(e) => setConfig((c) => ({ ...c, duration_s: Number(e.target.value) }))} />
          </div>

          <SectionLabel text="Tank" />
          <div>
            <label className="form-label">Volume [L]</label>
            <input className="form-input" type="number" step="1" value={config.tank.volume * 1000}
              onChange={(e) => setTank({ volume: Number(e.target.value) / 1000 })} />
          </div>
          <div>
            <label className="form-label">Initial Liquid Mass [kg]</label>
            <input className="form-input" type="number" step="1" value={config.tank.initial_liquid_mass}
              onChange={(e) => setTank({ initial_liquid_mass: Number(e.target.value) })} />
          </div>
          <div>
            <label className="form-label">Initial Ullage Pressure [bar]</label>
            <input className="form-input" type="number" step="1" value={config.tank.initial_ullage_pressure / 1e5}
              onChange={(e) => setTank({ initial_ullage_pressure: Number(e.target.value) * 1e5 })} />
          </div>
          <div>
            <label className="form-label">Initial Temperature [K]</label>
            <input className="form-input" type="number" step="1" value={config.tank.initial_temperature}
              onChange={(e) => setTank({ initial_temperature: Number(e.target.value) })} />
          </div>

          <SectionLabel text="Pressurant" />
          <div>
            <label className="form-label">Supply Pressure [bar]</label>
            <input className="form-input" type="number" step="1" value={config.pressurant.supply_pressure / 1e5}
              onChange={(e) => setPres({ supply_pressure: Number(e.target.value) * 1e5 })} />
          </div>

          <SectionLabel text="Propellant Flow" />
          <div>
            <label className="form-label">Mass Flow Rate [kg/s]</label>
            <input className="form-input" type="number" step="0.05" value={config.propellant.mass_flow_rate}
              onChange={(e) => setProp({ mass_flow_rate: Number(e.target.value) })} />
          </div>
        </div>
      </div>

      <div className="app-workspace">
        <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
          <div className="workspace-tabs">
            {["pressure", "mass", "temperature"].map((tab) => (
              <button key={tab} className={`workspace-tab ${activeTab === tab ? "active" : ""}`} onClick={() => setActiveTab(tab)}>
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
          <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
            {!result && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-muted)", fontSize: 12 }}>
                Configure parameters and run the simulation
              </div>
            )}
            {result && activeTab === "pressure" && (
              <PlotlyRenderer figureJson={pressureFigure} loading={isRunning} height={420} />
            )}
            {result && activeTab === "mass" && (
              <PlotlyRenderer figureJson={massFigure} loading={isRunning} height={420} />
            )}
            {result && activeTab === "temperature" && (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead><tr>
                  <th style={thStyle}>Time [s]</th>
                  <th style={thStyle}>Liquid T [K]</th>
                  <th style={thStyle}>Ullage T [K]</th>
                </tr></thead>
                <tbody>
                  {result.time_s.map((t, i) => (
                    <tr key={i}>
                      <td style={tdStyle}>{t.toFixed(2)}</td>
                      <td style={tdStyle}>{result.liquid_temperature_k[i].toFixed(2)}</td>
                      <td style={tdStyle}>{result.ullage_temperature_k[i].toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>

      <div className="app-right-panel">
        <div style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 12 }}>
            Simulation Summary
          </div>
          {result ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <MetricRow label="Burn Duration" value={`${result.burn_duration_s.toFixed(2)} s`} />
              <MetricRow label="Initial Pressure" value={`${result.pressure_bar[0].toFixed(1)} bar`} />
              <MetricRow label="Final Pressure" value={`${result.final_pressure_bar.toFixed(1)} bar`} />
              <MetricRow label="Initial Mass" value={`${result.liquid_mass_kg[0].toFixed(2)} kg`} />
              <MetricRow label="Final Mass" value={`${result.final_liquid_mass_kg.toFixed(2)} kg`} />
              <MetricRow label="Mass Expended" value={`${(result.liquid_mass_kg[0] - result.final_liquid_mass_kg).toFixed(2)} kg`} />
            </div>
          ) : (
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>No results yet</div>
          )}
        </div>
      </div>

      <footer className="app-statusbar">
        <div className="statusbar-item">
          <Icon icon={isRunning ? "dot" : errorMsg ? "cross" : result ? "tick-circle" : "circle"} size={10} />
          <span>{isRunning ? "SIMULATING" : errorMsg ? "ERROR" : result ? "NOMINAL" : "READY"}</span>
        </div>
        {errorMsg && <div className="statusbar-item" style={{ color: "var(--red)" }}>{errorMsg.slice(0, 80)}</div>}
        <div className="statusbar-spacer" />
        <div className="statusbar-item"><Icon icon="flame" size={10} /> RESA v2.0.0</div>
      </footer>
    </>
  );
}
