import { useState, useCallback } from "react";
import { Icon } from "@blueprintjs/core";
import { ModuleGate } from "../../components/common/ModuleGate";
import { useDesignSessionStore } from "../../store/designSessionStore";
import { usePerformanceMapsMutation } from "../../api/performance";
import type { PerformanceMapConfig, PerformanceFullResponse } from "../../types/performance";
import { DEFAULT_PERFORMANCE_CONFIG } from "../../types/performance";

function extractError(err: unknown): string {
  if (err && typeof err === "object" && "response" in err) {
    const resp = (err as { response?: { data?: { detail?: unknown } } }).response;
    if (typeof resp?.data?.detail === "string") return resp.data.detail;
  }
  if (err instanceof Error) return err.message;
  return "Performance analysis failed.";
}

function PerformanceConfigForm({
  config,
  onChange,
  onRun,
  isRunning,
}: {
  config: PerformanceMapConfig;
  onChange: (p: Partial<PerformanceMapConfig>) => void;
  onRun: () => void;
  isRunning: boolean;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)" }}>
        Altitude Sweep
      </div>
      <div>
        <label className="form-label">Min Altitude [km]</label>
        <input className="form-input" type="number" step="1" value={config.altitude_range_min_m / 1000} onChange={(e) => onChange({ altitude_range_min_m: Number(e.target.value) * 1000 })} />
      </div>
      <div>
        <label className="form-label">Max Altitude [km]</label>
        <input className="form-input" type="number" step="10" value={config.altitude_range_max_m / 1000} onChange={(e) => onChange({ altitude_range_max_m: Number(e.target.value) * 1000 })} />
      </div>
      <div>
        <label className="form-label">Points</label>
        <input className="form-input" type="number" step="10" value={config.altitude_points} onChange={(e) => onChange({ altitude_points: Number(e.target.value) })} />
      </div>

      <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginTop: 8 }}>
        Throttle Sweep
      </div>
      <div>
        <label className="form-label">Min Throttle [%]</label>
        <input className="form-input" type="number" step="5" value={config.throttle_range_min * 100} onChange={(e) => onChange({ throttle_range_min: Number(e.target.value) / 100 })} />
      </div>
      <div>
        <label className="form-label">Max Throttle [%]</label>
        <input className="form-input" type="number" step="5" value={config.throttle_range_max * 100} onChange={(e) => onChange({ throttle_range_max: Number(e.target.value) / 100 })} />
      </div>
      <div>
        <label className="form-label">Points</label>
        <input className="form-input" type="number" step="5" value={config.throttle_points} onChange={(e) => onChange({ throttle_points: Number(e.target.value) })} />
      </div>

      <button className={`run-btn ${isRunning ? "is-running" : ""}`} onClick={onRun} disabled={isRunning} style={{ marginTop: 12, width: "100%" }}>
        <Icon icon={isRunning ? "dot" : "play"} size={12} />
        {isRunning ? "COMPUTING..." : "RUN ANALYSIS"}
      </button>
    </div>
  );
}

function sampleIndices(total: number, count: number): number[] {
  if (total <= count) return Array.from({ length: total }, (_, i) => i);
  const step = (total - 1) / (count - 1);
  return Array.from({ length: count }, (_, i) => Math.round(i * step));
}

const thStyle: React.CSSProperties = { textAlign: "left", padding: "4px 8px", borderBottom: "1px solid var(--border-subtle)", color: "var(--text-muted)", fontWeight: 600 };
const tdStyle: React.CSSProperties = { padding: "3px 8px", fontFamily: "var(--font-mono)", borderBottom: "1px solid var(--border-subtle)" };

function MetricRow({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
      <span style={{ color: "var(--text-muted)" }}>{label}</span>
      <span style={{ fontFamily: "var(--font-mono)", color: warn ? "#e65100" : "var(--text-primary)" }}>{value}</span>
    </div>
  );
}

export default function PerformancePage() {
  const { sessionId, markModuleCompleted } = useDesignSessionStore();
  const [config, setConfig] = useState<PerformanceMapConfig>({ ...DEFAULT_PERFORMANCE_CONFIG });
  const [result, setResult] = useState<PerformanceFullResponse | null>(null);
  const [activeTab, setActiveTab] = useState("altitude");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const mutation = usePerformanceMapsMutation();
  const isRunning = mutation.isPending;

  const handleRun = useCallback(async () => {
    if (!sessionId) return;
    setErrorMsg(null);
    try {
      const res = await mutation.mutateAsync({ sessionId, config });
      setResult(res);
      markModuleCompleted("performance");
    } catch (err) {
      setErrorMsg(extractError(err));
    }
  }, [sessionId, config, mutation, markModuleCompleted]);

  const alt = result?.altitude;
  const thr = result?.throttle;

  return (
    <ModuleGate requires={["engine"]}>
      <header className="app-topbar">
        <div className="topbar-logo"><div className="topbar-logo-dot" />RESA</div>
        <div className="topbar-breadcrumb"><span className="page-name">Performance Maps</span></div>
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
            Configuration
          </span>
        </div>
        <div className="panel-scroll" style={{ padding: "12px 16px" }}>
          <PerformanceConfigForm config={config} onChange={(p) => setConfig((c) => ({ ...c, ...p }))} onRun={handleRun} isRunning={isRunning} />
        </div>
      </div>

      <div className="app-workspace">
        <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
          <div className="workspace-tabs">
            {["altitude", "throttle"].map((tab) => (
              <button key={tab} className={`workspace-tab ${activeTab === tab ? "active" : ""}`} onClick={() => setActiveTab(tab)}>
                {tab === "altitude" ? "Altitude Curve" : "Throttle Map"}
              </button>
            ))}
          </div>
          <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
            {!result && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-muted)", fontSize: 12 }}>
                Configure parameters and run analysis
              </div>
            )}

            {activeTab === "altitude" && alt && (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Alt [km]</th>
                    <th style={thStyle}>Thrust [N]</th>
                    <th style={thStyle}>Isp [s]</th>
                    <th style={thStyle}>Cf</th>
                  </tr>
                </thead>
                <tbody>
                  {sampleIndices(alt.altitudes_m.length, 20).map((i) => (
                    <tr key={i}>
                      <td style={tdStyle}>{(alt.altitudes_m[i] / 1000).toFixed(1)}</td>
                      <td style={tdStyle}>{alt.thrust_n[i].toFixed(1)}</td>
                      <td style={tdStyle}>{alt.isp_s[i].toFixed(1)}</td>
                      <td style={tdStyle}>{alt.cf[i].toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {activeTab === "throttle" && thr && (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Throttle [%]</th>
                    <th style={thStyle}>Pc [bar]</th>
                    <th style={thStyle}>Thrust [N]</th>
                    <th style={thStyle}>Isp [s]</th>
                  </tr>
                </thead>
                <tbody>
                  {thr.throttle_pcts.map((pct, i) => (
                    <tr key={i}>
                      <td style={tdStyle}>{pct.toFixed(0)}</td>
                      <td style={tdStyle}>{thr.pc_bar[i].toFixed(1)}</td>
                      <td style={tdStyle}>{thr.thrust_n[i].toFixed(1)}</td>
                      <td style={tdStyle}>{thr.isp_s[i].toFixed(1)}</td>
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
            Performance Summary
          </div>
          {alt ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <MetricRow label="Sea Level Thrust" value={`${alt.thrust_n[0].toFixed(0)} N`} />
              <MetricRow label="Vacuum Thrust" value={`${alt.thrust_n[alt.thrust_n.length - 1].toFixed(0)} N`} />
              <MetricRow label="Sea Level Isp" value={`${alt.isp_s[0].toFixed(1)} s`} />
              <MetricRow label="Vacuum Isp" value={`${alt.isp_s[alt.isp_s.length - 1].toFixed(1)} s`} />
              {alt.separation_altitude_m != null && (
                <MetricRow label="Separation Alt" value={`${(alt.separation_altitude_m / 1000).toFixed(1)} km`} />
              )}
            </div>
          ) : (
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>No results yet</div>
          )}
          {thr && (
            <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 16 }}>
              <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 4 }}>
                Throttle
              </div>
              <MetricRow label="Min Thrust" value={`${Math.min(...thr.thrust_n).toFixed(0)} N`} />
              <MetricRow label="Max Thrust" value={`${Math.max(...thr.thrust_n).toFixed(0)} N`} />
              <MetricRow label="Throttle Ratio" value={`${(Math.max(...thr.thrust_n) / Math.min(...thr.thrust_n)).toFixed(1)}:1`} />
            </div>
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
