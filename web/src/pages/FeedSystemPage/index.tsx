import { useState, useCallback } from "react";
import { Icon } from "@blueprintjs/core";
import { ModuleGate } from "../../components/common/ModuleGate";
import { useDesignSessionStore } from "../../store/designSessionStore";
import { useFeedSystemMutation } from "../../api/feed_system";
import type { FeedSystemConfig, FeedSystemResponse } from "../../types/feed_system";
import { DEFAULT_FEED_SYSTEM_CONFIG } from "../../types/feed_system";

function extractError(err: unknown): string {
  if (err && typeof err === "object" && "response" in err) {
    const resp = (err as { response?: { data?: { detail?: unknown } } }).response;
    if (typeof resp?.data?.detail === "string") return resp.data.detail;
  }
  if (err instanceof Error) return err.message;
  return "Feed system analysis failed.";
}

function FeedSystemConfigForm({
  config,
  onChange,
  onRun,
  isRunning,
}: {
  config: FeedSystemConfig;
  onChange: (p: Partial<FeedSystemConfig>) => void;
  onRun: () => void;
  isRunning: boolean;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div>
        <label className="form-label">Feed Type</label>
        <select className="form-select" value={config.feed_type} onChange={(e) => onChange({ feed_type: e.target.value as "pressure-fed" | "pump-fed" })}>
          <option value="pressure-fed">Pressure-Fed</option>
          <option value="pump-fed">Pump-Fed</option>
        </select>
      </div>

      {config.feed_type === "pump-fed" && (
        <div>
          <label className="form-label">Cycle Type</label>
          <select className="form-select" value={config.cycle_type} onChange={(e) => onChange({ cycle_type: e.target.value as FeedSystemConfig["cycle_type"] })}>
            <option value="none">None (Electric)</option>
            <option value="gas-generator">Gas Generator</option>
            <option value="expander">Expander</option>
            <option value="staged-combustion">Staged Combustion</option>
          </select>
        </div>
      )}

      <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginTop: 8 }}>
        Oxidizer Line
      </div>
      <div>
        <label className="form-label">Length [m]</label>
        <input className="form-input" type="number" step="0.1" value={config.ox_line_length_m} onChange={(e) => onChange({ ox_line_length_m: Number(e.target.value) })} />
      </div>
      <div>
        <label className="form-label">Diameter [mm]</label>
        <input className="form-input" type="number" step="1" value={config.ox_line_diameter_m * 1000} onChange={(e) => onChange({ ox_line_diameter_m: Number(e.target.value) / 1000 })} />
      </div>
      <div>
        <label className="form-label">K Fittings</label>
        <input className="form-input" type="number" step="1" value={config.ox_k_fittings} onChange={(e) => onChange({ ox_k_fittings: Number(e.target.value) })} />
      </div>

      <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginTop: 8 }}>
        Fuel Line
      </div>
      <div>
        <label className="form-label">Length [m]</label>
        <input className="form-input" type="number" step="0.1" value={config.fuel_line_length_m} onChange={(e) => onChange({ fuel_line_length_m: Number(e.target.value) })} />
      </div>
      <div>
        <label className="form-label">Diameter [mm]</label>
        <input className="form-input" type="number" step="1" value={config.fuel_line_diameter_m * 1000} onChange={(e) => onChange({ fuel_line_diameter_m: Number(e.target.value) / 1000 })} />
      </div>
      <div>
        <label className="form-label">K Fittings</label>
        <input className="form-input" type="number" step="1" value={config.fuel_k_fittings} onChange={(e) => onChange({ fuel_k_fittings: Number(e.target.value) })} />
      </div>

      {config.feed_type === "pump-fed" && (
        <>
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginTop: 8 }}>
            Pump
          </div>
          <div>
            <label className="form-label">Pump Efficiency</label>
            <input className="form-input" type="number" step="0.05" value={config.pump_efficiency} onChange={(e) => onChange({ pump_efficiency: Number(e.target.value) })} />
          </div>
          {config.cycle_type === "gas-generator" && (
            <>
              <div>
                <label className="form-label">GG Temperature [K]</label>
                <input className="form-input" type="number" step="50" value={config.gg_temperature_k} onChange={(e) => onChange({ gg_temperature_k: Number(e.target.value) })} />
              </div>
              <div>
                <label className="form-label">GG MR (fuel-rich)</label>
                <input className="form-input" type="number" step="0.05" value={config.gg_mr} onChange={(e) => onChange({ gg_mr: Number(e.target.value) })} />
              </div>
            </>
          )}
        </>
      )}

      <button className={`run-btn ${isRunning ? "is-running" : ""}`} onClick={onRun} disabled={isRunning} style={{ marginTop: 12, width: "100%" }}>
        <Icon icon={isRunning ? "dot" : "play"} size={12} />
        {isRunning ? "COMPUTING..." : "RUN ANALYSIS"}
      </button>
    </div>
  );
}

function MetricRow({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
      <span style={{ color: "var(--text-muted)" }}>{label}</span>
      <span style={{ fontFamily: "var(--font-mono)", color: warn ? "#e65100" : "var(--text-primary)" }}>{value}</span>
    </div>
  );
}

export default function FeedSystemPage() {
  const { sessionId, markModuleCompleted } = useDesignSessionStore();
  const [config, setConfig] = useState<FeedSystemConfig>({ ...DEFAULT_FEED_SYSTEM_CONFIG });
  const [result, setResult] = useState<FeedSystemResponse | null>(null);
  const [activeTab, setActiveTab] = useState("budget");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const mutation = useFeedSystemMutation();
  const isRunning = mutation.isPending;

  const handleRun = useCallback(async () => {
    if (!sessionId) return;
    setErrorMsg(null);
    try {
      const res = await mutation.mutateAsync({ sessionId, config });
      setResult(res);
      markModuleCompleted("feed_system");
    } catch (err) {
      setErrorMsg(extractError(err));
    }
  }, [sessionId, config, mutation, markModuleCompleted]);

  return (
    <ModuleGate requires={["engine"]}>
      <header className="app-topbar">
        <div className="topbar-logo"><div className="topbar-logo-dot" />RESA</div>
        <div className="topbar-breadcrumb"><span className="page-name">Feed System</span></div>
        <div className="topbar-spacer" />
        <div className="topbar-actions">
          <button className={`run-btn ${isRunning ? "is-running" : ""}`} onClick={handleRun} disabled={isRunning}>
            <Icon icon={isRunning ? "dot" : "play"} size={12} />
            {isRunning ? "COMPUTING..." : "RUN ANALYSIS"}
          </button>
        </div>
      </header>

      <div className="app-left-panel">
        <div style={{ flexShrink: 0, padding: "7px 12px", borderBottom: "1px solid var(--border-subtle)" }}>
          <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)" }}>
            Feed System Config
          </span>
        </div>
        <div className="panel-scroll" style={{ padding: "12px 16px" }}>
          <FeedSystemConfigForm config={config} onChange={(p) => setConfig((c) => ({ ...c, ...p }))} onRun={handleRun} isRunning={isRunning} />
        </div>
      </div>

      <div className="app-workspace">
        <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
          <div className="workspace-tabs">
            {["budget", "details"].map((tab) => (
              <button key={tab} className={`workspace-tab ${activeTab === tab ? "active" : ""}`} onClick={() => setActiveTab(tab)}>
                {tab === "budget" ? "Pressure Budget" : "Circuit Details"}
              </button>
            ))}
          </div>
          <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
            {!result && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-muted)", fontSize: 12 }}>
                Configure feed system and run analysis
              </div>
            )}

            {result && activeTab === "budget" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)", marginBottom: 4 }}>
                  Pressure Waterfall [bar]
                </div>
                {/* Waterfall visualization */}
                {[
                  { label: "Chamber Pressure", value: 0, isBase: true },
                  { label: "Injector \u0394P", value: result.injector_dp_bar },
                  { label: "Cooling \u0394P", value: result.cooling_dp_bar },
                  { label: "Ox Line Losses", value: result.line_losses_ox_bar },
                  { label: "Fuel Line Losses", value: result.line_losses_fuel_bar },
                ].map((item, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ width: 120, fontSize: 11, color: "var(--text-muted)" }}>{item.label}</span>
                    <div style={{ flex: 1, height: 16, background: "var(--bg-panel)", borderRadius: 2, overflow: "hidden", position: "relative" }}>
                      <div style={{
                        width: `${Math.min((item.isBase ? result.required_feed_pressure_bar : item.value) / result.required_feed_pressure_bar * 100, 100)}%`,
                        height: "100%",
                        background: item.isBase ? "var(--accent-bright)" : "#2d5a8a",
                        borderRadius: 2,
                      }} />
                    </div>
                    <span style={{ width: 50, textAlign: "right", fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>
                      {item.isBase ? result.required_feed_pressure_bar.toFixed(1) : item.value.toFixed(2)}
                    </span>
                  </div>
                ))}
                <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: 8, display: "flex", justifyContent: "space-between", fontSize: 12, fontWeight: 600 }}>
                  <span style={{ color: "var(--text-muted)" }}>Required Feed Pressure</span>
                  <span style={{ fontFamily: "var(--font-mono)", color: "var(--accent-bright)" }}>{result.required_feed_pressure_bar.toFixed(1)} bar</span>
                </div>
              </div>
            )}

            {result && activeTab === "details" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div>
                  <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 6 }}>Feed Type</div>
                  <div style={{ fontSize: 12, color: "var(--text-primary)" }}>{result.feed_type}{result.cycle_type !== "none" ? ` / ${result.cycle_type}` : ""}</div>
                </div>
                {result.pump_power_ox_w > 0 && (
                  <div>
                    <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 6 }}>Pump Sizing</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      <MetricRow label="Ox Pump Power" value={`${(result.pump_power_ox_w / 1000).toFixed(2)} kW`} />
                      <MetricRow label="Fuel Pump Power" value={`${(result.pump_power_fuel_w / 1000).toFixed(2)} kW`} />
                      <MetricRow label="Ox Head" value={`${result.pump_head_ox_m.toFixed(1)} m`} />
                      <MetricRow label="Fuel Head" value={`${result.pump_head_fuel_m.toFixed(1)} m`} />
                      <MetricRow label="NPSHa" value={`${result.npsh_available_m.toFixed(1)} m`} />
                    </div>
                  </div>
                )}
                {result.turbine_power_w > 0 && (
                  <div>
                    <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 6 }}>Cycle Balance</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      <MetricRow label="Turbine Power" value={`${(result.turbine_power_w / 1000).toFixed(2)} kW`} />
                      <MetricRow label="Power Margin" value={`${result.power_balance_margin_pct.toFixed(1)}%`} warn={result.power_balance_margin_pct < 0} />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="app-right-panel">
        <div style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 12 }}>
            Feed System Summary
          </div>
          {result ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <MetricRow label="Feed Pressure" value={`${result.required_feed_pressure_bar.toFixed(1)} bar`} />
              <MetricRow label="Injector \u0394P" value={`${result.injector_dp_bar.toFixed(1)} bar`} />
              <MetricRow label="Cooling \u0394P" value={`${result.cooling_dp_bar.toFixed(1)} bar`} />
              <MetricRow label="Line Loss (Ox)" value={`${result.line_losses_ox_bar.toFixed(2)} bar`} />
              <MetricRow label="Line Loss (Fuel)" value={`${result.line_losses_fuel_bar.toFixed(2)} bar`} />
              {result.warnings.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  {result.warnings.map((w, i) => (
                    <div key={i} style={{ fontSize: 10, color: "#e65100", display: "flex", gap: 4, alignItems: "flex-start", marginBottom: 4 }}>
                      <Icon icon="warning-sign" size={10} />
                      <span>{w}</span>
                    </div>
                  ))}
                </div>
              )}
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
    </ModuleGate>
  );
}
