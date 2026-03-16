import { useState, useCallback, useMemo } from "react";
import { Icon } from "@blueprintjs/core";
import { ModuleGate } from "../../components/common/ModuleGate";
import { useDesignSessionStore } from "../../store/designSessionStore";
import { useOptimizationMutation } from "../../api/optimization";
import type { OptimizationConfig, OptimizationResponse, DesignVariableSpec } from "../../types/optimization";
import { DEFAULT_OPTIMIZATION_CONFIG } from "../../types/optimization";
import { PlotlyRenderer } from "../../components/plots/PlotlyRenderer";

function extractError(err: unknown): string {
  if (err && typeof err === "object" && "response" in err) {
    const resp = (err as { response?: { data?: { detail?: unknown } } }).response;
    if (typeof resp?.data?.detail === "string") return resp.data.detail;
  }
  if (err instanceof Error) return err.message;
  return "Optimization failed.";
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

const DEFAULT_VARIABLES: DesignVariableSpec[] = [
  { name: "pc_bar", min_val: 10, max_val: 50, initial: 25 },
  { name: "mr", min_val: 2.0, max_val: 8.0, initial: 4.0 },
];

const VARIABLE_OPTIONS = ["pc_bar", "mr", "thrust_n", "expansion_ratio", "eff_combustion"];
const OBJECTIVE_OPTIONS = ["isp_vac", "isp_sea", "thrust_vac", "thrust_sea", "massflow_total"];
const ALGORITHM_OPTIONS = ["Nelder-Mead", "SLSQP", "L-BFGS-B"];

export default function OptimizationPage() {
  const { sessionId } = useDesignSessionStore();
  const [config, setConfig] = useState<OptimizationConfig>({
    ...DEFAULT_OPTIMIZATION_CONFIG,
    variables: DEFAULT_VARIABLES.map((v) => ({ ...v })),
  });
  const [result, setResult] = useState<OptimizationResponse | null>(null);
  const [activeTab, setActiveTab] = useState("results");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const mutation = useOptimizationMutation();
  const isRunning = mutation.isPending;

  const handleRun = useCallback(async () => {
    if (!sessionId) return;
    setErrorMsg(null);
    try {
      const res = await mutation.mutateAsync({ sessionId, config });
      setResult(res);
    } catch (err) {
      setErrorMsg(extractError(err));
    }
  }, [sessionId, config, mutation]);

  const updateVar = (i: number, patch: Partial<DesignVariableSpec>) => {
    setConfig((c) => {
      const vars = [...c.variables];
      vars[i] = { ...vars[i], ...patch };
      return { ...c, variables: vars };
    });
  };

  const addVar = () => setConfig((c) => ({
    ...c,
    variables: [...c.variables, { name: "pc_bar", min_val: 10, max_val: 50, initial: 25 }],
  }));

  const removeVar = (i: number) => setConfig((c) => ({
    ...c, variables: c.variables.filter((_, idx) => idx !== i),
  }));

  const convergenceFigure = useMemo(() => {
    if (!result || result.history_iterations.length === 0) return null;
    return JSON.stringify({
      data: [{
        type: "scatter",
        mode: "lines+markers",
        x: result.history_iterations,
        y: result.history_objective,
        name: config.objective,
        line: { color: "#58a6ff", width: 2 },
        marker: { size: 4, color: "#58a6ff" },
      }],
      layout: {
        paper_bgcolor: "#0d1117",
        plot_bgcolor: "#0d1117",
        font: { color: "#c9d1d9", size: 11 },
        margin: { t: 24, r: 16, b: 48, l: 64 },
        xaxis: { gridcolor: "#21262d", zerolinecolor: "#21262d", title: "Iteration" },
        yaxis: { gridcolor: "#21262d", zerolinecolor: "#21262d", title: config.objective },
      },
    });
  }, [result, config.objective]);

  return (
    <ModuleGate requires={["engine"]}>
      <header className="app-topbar">
        <div className="topbar-logo"><div className="topbar-logo-dot" />RESA</div>
        <div className="topbar-breadcrumb"><span className="page-name">Optimization</span></div>
        <div className="topbar-spacer" />
        <div className="topbar-actions">
          <button className={`run-btn ${isRunning ? "is-running" : ""}`} onClick={handleRun} disabled={isRunning}>
            <Icon icon={isRunning ? "dot" : "play"} size={12} />
            {isRunning ? "OPTIMIZING..." : "RUN OPTIMIZATION"}
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
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 4 }}>
            Objective
          </div>
          <div>
            <label className="form-label">Maximize / Minimize</label>
            <select className="form-select" value={config.objective}
              onChange={(e) => setConfig((c) => ({ ...c, objective: e.target.value }))}>
              {OBJECTIVE_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </div>
          <div>
            <label className="form-label">Direction</label>
            <select className="form-select" value={config.minimize ? "minimize" : "maximize"}
              onChange={(e) => setConfig((c) => ({ ...c, minimize: e.target.value === "minimize" }))}>
              <option value="maximize">Maximize</option>
              <option value="minimize">Minimize</option>
            </select>
          </div>
          <div>
            <label className="form-label">Algorithm</label>
            <select className="form-select" value={config.algorithm}
              onChange={(e) => setConfig((c) => ({ ...c, algorithm: e.target.value }))}>
              {ALGORITHM_OPTIONS.map((a) => <option key={a} value={a}>{a}</option>)}
            </select>
          </div>
          <div>
            <label className="form-label">Max Iterations</label>
            <input className="form-input" type="number" step="10" min="5" value={config.max_iterations}
              onChange={(e) => setConfig((c) => ({ ...c, max_iterations: Number(e.target.value) }))} />
          </div>

          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginTop: 12, marginBottom: 4 }}>
            Design Variables
          </div>
          {config.variables.map((v, i) => (
            <div key={i} style={{ marginBottom: 8, padding: "8px", border: "1px solid var(--border-subtle)", borderRadius: 4 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                <select className="form-select" style={{ flex: 1, marginRight: 4 }} value={v.name}
                  onChange={(e) => updateVar(i, { name: e.target.value })}>
                  {VARIABLE_OPTIONS.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
                </select>
                <button onClick={() => removeVar(i)} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", padding: "0 4px" }}>×</button>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 4 }}>
                <div>
                  <label className="form-label">Min</label>
                  <input className="form-input" type="number" step="any" value={v.min_val}
                    onChange={(e) => updateVar(i, { min_val: Number(e.target.value) })} />
                </div>
                <div>
                  <label className="form-label">Initial</label>
                  <input className="form-input" type="number" step="any" value={v.initial}
                    onChange={(e) => updateVar(i, { initial: Number(e.target.value) })} />
                </div>
                <div>
                  <label className="form-label">Max</label>
                  <input className="form-input" type="number" step="any" value={v.max_val}
                    onChange={(e) => updateVar(i, { max_val: Number(e.target.value) })} />
                </div>
              </div>
            </div>
          ))}
          <button onClick={addVar} style={{ width: "100%", padding: "6px", background: "none", border: "1px dashed var(--border-subtle)", borderRadius: 4, cursor: "pointer", color: "var(--text-muted)", fontSize: 11 }}>
            + Add Variable
          </button>
        </div>
      </div>

      <div className="app-workspace">
        <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
          <div className="workspace-tabs">
            {["results", "convergence"].map((tab) => (
              <button key={tab} className={`workspace-tab ${activeTab === tab ? "active" : ""}`} onClick={() => setActiveTab(tab)}>
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
          <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
            {!result && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-muted)", fontSize: 12 }}>
                Configure variables and run the optimization
              </div>
            )}
            {result && activeTab === "results" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div>
                  <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 6, color: "var(--text-secondary)" }}>Optimal Variables</div>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                    <thead><tr><th style={thStyle}>Variable</th><th style={thStyle}>Optimal Value</th></tr></thead>
                    <tbody>
                      {Object.entries(result.optimal_variables).map(([k, v]) => (
                        <tr key={k}><td style={tdStyle}>{k}</td><td style={tdStyle}>{v.toFixed(4)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div>
                  <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 6, color: "var(--text-secondary)" }}>Optimal Outputs</div>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                    <thead><tr><th style={thStyle}>Output</th><th style={thStyle}>Value</th></tr></thead>
                    <tbody>
                      {Object.entries(result.optimal_outputs).map(([k, v]) => (
                        <tr key={k}><td style={tdStyle}>{k}</td><td style={tdStyle}>{v.toFixed(4)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            {result && activeTab === "convergence" && (
              <PlotlyRenderer figureJson={convergenceFigure} loading={isRunning} height={420} />
            )}
          </div>
        </div>
      </div>

      <div className="app-right-panel">
        <div style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 12 }}>
            Optimization Summary
          </div>
          {result ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <MetricRow label="Converged" value={result.converged ? "Yes" : "No"} />
              <MetricRow label="Evaluations" value={`${result.n_evaluations}`} />
              <MetricRow label={`Best ${config.objective}`} value={result.objective_value.toFixed(4)} />
              <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 8, wordBreak: "break-word" }}>{result.message}</div>
            </div>
          ) : (
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>No results yet</div>
          )}
        </div>
      </div>

      <footer className="app-statusbar">
        <div className="statusbar-item">
          <Icon icon={isRunning ? "dot" : errorMsg ? "cross" : result ? (result.converged ? "tick-circle" : "warning-sign") : "circle"} size={10} />
          <span>{isRunning ? "OPTIMIZING" : errorMsg ? "ERROR" : result ? (result.converged ? "CONVERGED" : "INCOMPLETE") : "READY"}</span>
        </div>
        {errorMsg && <div className="statusbar-item" style={{ color: "var(--red)" }}>{errorMsg.slice(0, 80)}</div>}
        <div className="statusbar-spacer" />
        <div className="statusbar-item"><Icon icon="flame" size={10} /> RESA v2.0.0</div>
      </footer>
    </ModuleGate>
  );
}
