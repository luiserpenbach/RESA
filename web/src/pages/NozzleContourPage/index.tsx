import { useState, useCallback } from "react";
import { Icon } from "@blueprintjs/core";
import { ModuleGate } from "../../components/common/ModuleGate";
import { useDesignSessionStore } from "../../store/designSessionStore";
import { useGenerateContourMutation, exportContour } from "../../api/contour";
import type { ContourConfig, ContourResponse } from "../../types/contour";
import { DEFAULT_CONTOUR_CONFIG } from "../../types/contour";

function extractError(err: unknown): string {
  if (err && typeof err === "object" && "response" in err) {
    const resp = (err as { response?: { data?: { detail?: unknown } } }).response;
    if (typeof resp?.data?.detail === "string") return resp.data.detail;
  }
  if (err instanceof Error) return err.message;
  return "Contour generation failed.";
}

function ContourConfigForm({
  config,
  onChange,
  onRun,
  onExport,
  isRunning,
  hasResult,
}: {
  config: ContourConfig;
  onChange: (partial: Partial<ContourConfig>) => void;
  onRun: () => void;
  onExport: (fmt: string) => void;
  isRunning: boolean;
  hasResult: boolean;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div>
        <label className="form-label">Bell Fraction</label>
        <input
          className="form-input"
          type="number"
          step="0.05"
          placeholder="From engine config"
          value={config.bell_fraction ?? ""}
          onChange={(e) =>
            onChange({ bell_fraction: e.target.value ? Number(e.target.value) : null })
          }
        />
      </div>

      <div>
        <label className="form-label">Exit Angle [deg]</label>
        <input
          className="form-input"
          type="number"
          step="1"
          placeholder="From engine config"
          value={config.theta_exit ?? ""}
          onChange={(e) =>
            onChange({ theta_exit: e.target.value ? Number(e.target.value) : null })
          }
        />
      </div>

      <div>
        <label className="form-label">Resolution [points]</label>
        <input
          className="form-input"
          type="number"
          step="50"
          value={config.resolution}
          onChange={(e) => onChange({ resolution: Number(e.target.value) })}
        />
      </div>

      <div>
        <label className="form-label">Wall Thickness [mm]</label>
        <input
          className="form-input"
          type="number"
          step="0.1"
          value={config.wall_thickness_mm}
          onChange={(e) => onChange({ wall_thickness_mm: Number(e.target.value) })}
        />
      </div>

      <button
        className={`run-btn ${isRunning ? "is-running" : ""}`}
        onClick={onRun}
        disabled={isRunning}
        style={{ marginTop: 12, width: "100%" }}
      >
        <Icon icon={isRunning ? "dot" : "play"} size={12} />
        {isRunning ? "GENERATING..." : "GENERATE CONTOUR"}
      </button>

      {hasResult && (
        <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
          <button className="export-btn" onClick={() => onExport("csv")} style={exportBtnStyle}>
            <Icon icon="download" size={10} /> CSV
          </button>
          <button className="export-btn" onClick={() => onExport("json")} style={exportBtnStyle}>
            <Icon icon="download" size={10} /> JSON
          </button>
        </div>
      )}
    </div>
  );
}

const exportBtnStyle: React.CSSProperties = {
  flex: 1,
  padding: "4px 8px",
  fontSize: 10,
  border: "1px solid var(--border-subtle)",
  borderRadius: 3,
  background: "transparent",
  color: "var(--text-secondary)",
  cursor: "pointer",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  gap: 4,
};

function ContourWorkspace({
  result,
  activeTab,
  onTabChange,
}: {
  result: ContourResponse | null;
  activeTab: string;
  onTabChange: (t: string) => void;
}) {
  const tabs = ["2d", "3d", "data"];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div className="workspace-tabs">
        {tabs.map((tab) => (
          <button
            key={tab}
            className={`workspace-tab ${activeTab === tab ? "active" : ""}`}
            onClick={() => onTabChange(tab)}
          >
            {tab === "2d" ? "2D Contour" : tab === "3d" ? "3D View" : "Coordinates"}
          </button>
        ))}
      </div>

      <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
        {!result && (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-muted)", fontSize: 12 }}>
            Generate a contour to see results
          </div>
        )}

        {result && activeTab === "2d" && result.figure_contour && (
          <div
            style={{ width: "100%", height: "100%" }}
            ref={(el) => {
              if (el && result.figure_contour) {
                try {
                  const Plotly = (window as unknown as { Plotly?: { newPlot: Function } }).Plotly;
                  if (Plotly) {
                    const fig = JSON.parse(result.figure_contour);
                    Plotly.newPlot(el, fig.data, { ...fig.layout, autosize: true, paper_bgcolor: "transparent", plot_bgcolor: "transparent" });
                  }
                } catch { /* ignore */ }
              }
            }}
          />
        )}

        {result && activeTab === "2d" && !result.figure_contour && (
          <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
            Contour generated: {result.x_mm.length} points, L={result.total_length_mm.toFixed(1)} mm
          </div>
        )}

        {result && activeTab === "3d" && result.figure_3d && (
          <div
            style={{ width: "100%", height: "100%" }}
            ref={(el) => {
              if (el && result.figure_3d) {
                try {
                  const Plotly = (window as unknown as { Plotly?: { newPlot: Function } }).Plotly;
                  if (Plotly) {
                    const fig = JSON.parse(result.figure_3d);
                    Plotly.newPlot(el, fig.data, { ...fig.layout, autosize: true });
                  }
                } catch { /* ignore */ }
              }
            }}
          />
        )}

        {result && activeTab === "3d" && !result.figure_3d && (
          <div style={{ fontSize: 12, color: "var(--text-muted)" }}>3D view not available</div>
        )}

        {result && activeTab === "data" && (
          <div style={{ fontSize: 11, maxHeight: "100%", overflow: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={thStyle}>x [mm]</th>
                  <th style={thStyle}>r [mm]</th>
                </tr>
              </thead>
              <tbody>
                {sampleIndices(result.x_mm.length, 30).map((i) => (
                  <tr key={i}>
                    <td style={tdStyle}>{result.x_mm[i].toFixed(2)}</td>
                    <td style={tdStyle}>{result.y_mm[i].toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

const thStyle: React.CSSProperties = { textAlign: "left", padding: "4px 8px", borderBottom: "1px solid var(--border-subtle)", color: "var(--text-muted)", fontWeight: 600 };
const tdStyle: React.CSSProperties = { padding: "3px 8px", fontFamily: "var(--font-mono)", borderBottom: "1px solid var(--border-subtle)" };

function sampleIndices(total: number, count: number): number[] {
  if (total <= count) return Array.from({ length: total }, (_, i) => i);
  const step = (total - 1) / (count - 1);
  return Array.from({ length: count }, (_, i) => Math.round(i * step));
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
      <span style={{ color: "var(--text-muted)" }}>{label}</span>
      <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>{value}</span>
    </div>
  );
}

export default function NozzleContourPage() {
  const { sessionId, markModuleCompleted } = useDesignSessionStore();
  const [config, setConfig] = useState<ContourConfig>({ ...DEFAULT_CONTOUR_CONFIG });
  const [result, setResult] = useState<ContourResponse | null>(null);
  const [activeTab, setActiveTab] = useState("2d");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const mutation = useGenerateContourMutation();
  const isRunning = mutation.isPending;

  const handleRun = useCallback(async () => {
    if (!sessionId) return;
    setErrorMsg(null);
    try {
      const res = await mutation.mutateAsync({ sessionId, config });
      setResult(res);
      markModuleCompleted("contour");
    } catch (err) {
      setErrorMsg(extractError(err));
    }
  }, [sessionId, config, mutation, markModuleCompleted]);

  const handleExport = useCallback(async (format: string) => {
    if (!sessionId) return;
    try {
      const blob = await exportContour(sessionId, format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `nozzle_contour.${format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch { /* silent */ }
  }, [sessionId]);

  return (
    <ModuleGate requires={["engine"]}>
      <header className="app-topbar">
        <div className="topbar-logo"><div className="topbar-logo-dot" />RESA</div>
        <div className="topbar-breadcrumb"><span className="page-name">Nozzle Contour</span></div>
        <div className="topbar-spacer" />
        <div className="topbar-actions">
          <button className={`run-btn ${isRunning ? "is-running" : ""}`} onClick={handleRun} disabled={isRunning}>
            <Icon icon={isRunning ? "dot" : "play"} size={12} />
            {isRunning ? "GENERATING..." : "GENERATE"}
          </button>
        </div>
      </header>

      <div className="app-left-panel">
        <div style={{ flexShrink: 0, padding: "7px 12px", borderBottom: "1px solid var(--border-subtle)" }}>
          <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)" }}>
            Contour Parameters
          </span>
        </div>
        <div className="panel-scroll" style={{ padding: "12px 16px" }}>
          <ContourConfigForm config={config} onChange={(p) => setConfig((c) => ({ ...c, ...p }))} onRun={handleRun} onExport={handleExport} isRunning={isRunning} hasResult={!!result} />
        </div>
      </div>

      <div className="app-workspace">
        <ContourWorkspace result={result} activeTab={activeTab} onTabChange={setActiveTab} />
      </div>

      <div className="app-right-panel">
        <div style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 12 }}>
            Dimensions
          </div>
          {result ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <MetricRow label="Throat Dia" value={`${result.throat_diameter_mm.toFixed(2)} mm`} />
              <MetricRow label="Exit Dia" value={`${result.exit_diameter_mm.toFixed(2)} mm`} />
              <MetricRow label="Length" value={`${result.total_length_mm.toFixed(1)} mm`} />
              <MetricRow label="Exp. Ratio" value={result.expansion_ratio.toFixed(2)} />
              <MetricRow label="Points" value={String(result.x_mm.length)} />
            </div>
          ) : (
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>No results yet</div>
          )}
        </div>
      </div>

      <footer className="app-statusbar">
        <div className="statusbar-item">
          <Icon icon={isRunning ? "dot" : errorMsg ? "cross" : result ? "tick-circle" : "circle"} size={10} />
          <span>{isRunning ? "GENERATING" : errorMsg ? "ERROR" : result ? "NOMINAL" : "READY"}</span>
        </div>
        {errorMsg && <div className="statusbar-item" style={{ color: "var(--red)" }}>{errorMsg.slice(0, 80)}</div>}
        <div className="statusbar-spacer" />
        <div className="statusbar-item"><Icon icon="flame" size={10} /> RESA v2.0.0</div>
      </footer>
    </ModuleGate>
  );
}
