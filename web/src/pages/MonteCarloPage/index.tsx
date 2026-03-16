import { useState, useCallback, useMemo } from "react";
import { Icon } from "@blueprintjs/core";
import { ModuleGate } from "../../components/common/ModuleGate";
import { useDesignSessionStore } from "../../store/designSessionStore";
import { useMonteCarloMutation } from "../../api/monte_carlo";
import type { MonteCarloConfig, MonteCarloResponse, ParameterSpec } from "../../types/monte_carlo";
import { DEFAULT_MC_CONFIG } from "../../types/monte_carlo";
import { PlotlyRenderer } from "../../components/plots/PlotlyRenderer";

function extractError(err: unknown): string {
  if (err && typeof err === "object" && "response" in err) {
    const resp = (err as { response?: { data?: { detail?: unknown } } }).response;
    if (typeof resp?.data?.detail === "string") return resp.data.detail;
  }
  if (err instanceof Error) return err.message;
  return "Monte Carlo analysis failed.";
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

const DEFAULT_PARAMS: ParameterSpec[] = [
  { name: "pc_bar", nominal: 25.0, distribution: "normal", std_dev: 0.75, min_val: null, max_val: null, mode: null },
  { name: "mr", nominal: 4.0, distribution: "normal", std_dev: 0.12, min_val: null, max_val: null, mode: null },
];

const DARK_LAYOUT: object = {
  paper_bgcolor: "#0d1117",
  plot_bgcolor: "#0d1117",
  font: { color: "#c9d1d9", size: 11 },
  margin: { t: 24, r: 16, b: 48, l: 56 },
  xaxis: { gridcolor: "#21262d", zerolinecolor: "#21262d" },
  yaxis: { gridcolor: "#21262d", zerolinecolor: "#21262d" },
};

const TRACE_COLORS = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff"];

export default function MonteCarloPage() {
  const { sessionId } = useDesignSessionStore();
  const [config, setConfig] = useState<MonteCarloConfig>({
    ...DEFAULT_MC_CONFIG,
    parameters: DEFAULT_PARAMS.map((p) => ({ ...p })),
  });
  const [result, setResult] = useState<MonteCarloResponse | null>(null);
  const [activeTab, setActiveTab] = useState("statistics");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const mutation = useMonteCarloMutation();
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

  const updateParam = (i: number, patch: Partial<ParameterSpec>) => {
    setConfig((c) => {
      const params = [...c.parameters];
      params[i] = { ...params[i], ...patch };
      return { ...c, parameters: params };
    });
  };

  const addParam = () => setConfig((c) => ({
    ...c,
    parameters: [...c.parameters, { name: "pc_bar", nominal: 25.0, distribution: "normal", std_dev: 0.75, min_val: null, max_val: null, mode: null }],
  }));

  const removeParam = (i: number) => setConfig((c) => ({
    ...c, parameters: c.parameters.filter((_, idx) => idx !== i),
  }));

  const outputNames = result ? Object.keys(result.statistics) : [];

  const histogramFigure = useMemo(() => {
    if (!result) return null;
    const samples = result.output_samples;
    const names = Object.keys(samples);
    if (names.length === 0) return null;
    return JSON.stringify({
      data: names.map((name, i) => ({
        type: "histogram",
        x: samples[name],
        name,
        opacity: 0.75,
        marker: { color: TRACE_COLORS[i % TRACE_COLORS.length] },
        nbinsx: 30,
      })),
      layout: {
        ...DARK_LAYOUT,
        barmode: "overlay",
        xaxis: { ...(DARK_LAYOUT as Record<string, object>).xaxis, title: "Value" },
        yaxis: { ...(DARK_LAYOUT as Record<string, object>).yaxis, title: "Count" },
        legend: { bgcolor: "transparent" },
      },
    });
  }, [result]);

  const tornadoFigure = useMemo(() => {
    if (!result || outputNames.length === 0) return null;
    // Build tornado for the first output
    const outName = outputNames[0];
    const sens = result.sensitivity[outName] || {};
    const entries = Object.entries(sens).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const params = entries.map(([p]) => p);
    const values = entries.map(([, v]) => v);
    const colors = values.map((v) => (v >= 0 ? "#58a6ff" : "#f85149"));
    return JSON.stringify({
      data: [{
        type: "bar",
        orientation: "h",
        x: values,
        y: params,
        marker: { color: colors },
        name: "Pearson r",
      }],
      layout: {
        ...DARK_LAYOUT,
        title: { text: `Sensitivity — ${outName}`, font: { size: 12, color: "#8b949e" } },
        xaxis: { ...(DARK_LAYOUT as Record<string, object>).xaxis, title: "Pearson r", range: [-1, 1] },
        yaxis: { ...(DARK_LAYOUT as Record<string, object>).yaxis },
      },
    });
  }, [result, outputNames]);

  return (
    <ModuleGate requires={["engine"]}>
      <header className="app-topbar">
        <div className="topbar-logo"><div className="topbar-logo-dot" />RESA</div>
        <div className="topbar-breadcrumb"><span className="page-name">Monte Carlo Analysis</span></div>
        <div className="topbar-spacer" />
        <div className="topbar-actions">
          <button className={`run-btn ${isRunning ? "is-running" : ""}`} onClick={handleRun} disabled={isRunning}>
            <Icon icon={isRunning ? "dot" : "play"} size={12} />
            {isRunning ? "RUNNING..." : "RUN ANALYSIS"}
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
          <div>
            <label className="form-label">Samples</label>
            <input className="form-input" type="number" step="50" min="10" value={config.n_samples}
              onChange={(e) => setConfig((c) => ({ ...c, n_samples: Number(e.target.value) }))} />
          </div>

          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginTop: 12, marginBottom: 4 }}>
            Uncertain Parameters
          </div>
          {config.parameters.map((p, i) => (
            <div key={i} style={{ marginBottom: 10, padding: "8px", border: "1px solid var(--border-subtle)", borderRadius: 4 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                <select className="form-select" style={{ flex: 1, marginRight: 4 }} value={p.name}
                  onChange={(e) => updateParam(i, { name: e.target.value })}>
                  <option value="pc_bar">pc_bar</option>
                  <option value="mr">mr</option>
                  <option value="thrust_n">thrust_n</option>
                  <option value="eff_combustion">eff_combustion</option>
                  <option value="expansion_ratio">expansion_ratio</option>
                </select>
                <button onClick={() => removeParam(i)} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", padding: "0 4px" }}>×</button>
              </div>
              <div>
                <label className="form-label">Nominal</label>
                <input className="form-input" type="number" step="any" value={p.nominal}
                  onChange={(e) => updateParam(i, { nominal: Number(e.target.value) })} />
              </div>
              <div>
                <label className="form-label">Distribution</label>
                <select className="form-select" value={p.distribution}
                  onChange={(e) => updateParam(i, { distribution: e.target.value })}>
                  <option value="normal">Normal</option>
                  <option value="uniform">Uniform</option>
                  <option value="triangular">Triangular</option>
                </select>
              </div>
              {p.distribution === "normal" && (
                <div>
                  <label className="form-label">Std Dev</label>
                  <input className="form-input" type="number" step="any" value={p.std_dev ?? ""}
                    onChange={(e) => updateParam(i, { std_dev: Number(e.target.value) })} />
                </div>
              )}
              {(p.distribution === "uniform" || p.distribution === "triangular") && (
                <>
                  <div>
                    <label className="form-label">Min</label>
                    <input className="form-input" type="number" step="any" value={p.min_val ?? ""}
                      onChange={(e) => updateParam(i, { min_val: Number(e.target.value) })} />
                  </div>
                  <div>
                    <label className="form-label">Max</label>
                    <input className="form-input" type="number" step="any" value={p.max_val ?? ""}
                      onChange={(e) => updateParam(i, { max_val: Number(e.target.value) })} />
                  </div>
                </>
              )}
            </div>
          ))}
          <button onClick={addParam} style={{ width: "100%", padding: "6px", background: "none", border: "1px dashed var(--border-subtle)", borderRadius: 4, cursor: "pointer", color: "var(--text-muted)", fontSize: 11 }}>
            + Add Parameter
          </button>
        </div>
      </div>

      <div className="app-workspace">
        <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
          <div className="workspace-tabs">
            {["statistics", "sensitivity", "histogram"].map((tab) => (
              <button key={tab} className={`workspace-tab ${activeTab === tab ? "active" : ""}`} onClick={() => setActiveTab(tab)}>
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
          <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
            {!result && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-muted)", fontSize: 12 }}>
                Configure parameters and run the analysis
              </div>
            )}
            {result && activeTab === "statistics" && (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead><tr>
                  <th style={thStyle}>Output</th>
                  <th style={thStyle}>Mean</th>
                  <th style={thStyle}>Std</th>
                  <th style={thStyle}>P5</th>
                  <th style={thStyle}>P50</th>
                  <th style={thStyle}>P95</th>
                </tr></thead>
                <tbody>
                  {outputNames.map((name) => {
                    const s = result.statistics[name];
                    return (
                      <tr key={name}>
                        <td style={tdStyle}>{name}</td>
                        <td style={tdStyle}>{s.mean.toFixed(3)}</td>
                        <td style={tdStyle}>{s.std.toFixed(3)}</td>
                        <td style={tdStyle}>{s.p5.toFixed(3)}</td>
                        <td style={tdStyle}>{s.p50.toFixed(3)}</td>
                        <td style={tdStyle}>{s.p95.toFixed(3)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
            {result && activeTab === "sensitivity" && (
              <PlotlyRenderer figureJson={tornadoFigure} loading={isRunning} height={360} />
            )}
            {result && activeTab === "histogram" && (
              <PlotlyRenderer figureJson={histogramFigure} loading={isRunning} height={400} />
            )}
          </div>
        </div>
      </div>

      <div className="app-right-panel">
        <div style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 12 }}>
            Run Summary
          </div>
          {result ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <MetricRow label="Samples Run" value={`${result.n_samples}`} />
              <MetricRow label="Failed" value={`${result.n_failed}`} />
              <MetricRow label="Elapsed" value={`${result.elapsed_s.toFixed(1)} s`} />
              {outputNames.slice(0, 3).map((name) => {
                const s = result.statistics[name];
                return (
                  <div key={name} style={{ marginTop: 8 }}>
                    <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4 }}>{name}</div>
                    <MetricRow label="Mean" value={s.mean.toFixed(3)} />
                    <MetricRow label="P5–P95" value={`${s.p5.toFixed(2)} – ${s.p95.toFixed(2)}`} />
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>No results yet</div>
          )}
        </div>
      </div>

      <footer className="app-statusbar">
        <div className="statusbar-item">
          <Icon icon={isRunning ? "dot" : errorMsg ? "cross" : result ? "tick-circle" : "circle"} size={10} />
          <span>{isRunning ? "RUNNING" : errorMsg ? "ERROR" : result ? "NOMINAL" : "READY"}</span>
        </div>
        {errorMsg && <div className="statusbar-item" style={{ color: "var(--red)" }}>{errorMsg.slice(0, 80)}</div>}
        <div className="statusbar-spacer" />
        <div className="statusbar-item"><Icon icon="flame" size={10} /> RESA v2.0.0</div>
      </footer>
    </ModuleGate>
  );
}
