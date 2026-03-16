import React, { useState, useCallback, useEffect } from "react";
import { Icon } from "@blueprintjs/core";
import { ModuleGate } from "../../components/common/ModuleGate";
import { StaleDataBanner } from "../../components/common/StaleDataBanner";
import { useDesignSessionStore } from "../../store/designSessionStore";
import { useWallThicknessMutation, listMaterials } from "../../api/structural";
import type { WallThicknessConfig, WallThicknessResponse } from "../../types/structural";

function extractError(err: unknown): string {
  if (err && typeof err === "object" && "response" in err) {
    const resp = (err as { response?: { data?: { detail?: unknown } } }).response;
    const detail = resp?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  if (err instanceof Error) return err.message;
  return "Wall thickness analysis failed.";
}

const structInputStyle: React.CSSProperties = {
  width: "100%",
  height: 26,
  padding: "0 7px",
  background: "var(--bg-base)",
  border: "1px solid var(--border-default)",
  borderRadius: "var(--radius-sm)",
  color: "var(--text-primary)",
  fontFamily: "var(--font-mono)",
  fontSize: 12,
  boxSizing: "border-box" as const,
};

function StructuralConfigForm({
  config,
  onChange,
  onRun,
  isRunning,
  materials,
}: {
  config: WallThicknessConfig;
  onChange: (partial: Partial<WallThicknessConfig>) => void;
  onRun: () => void;
  isRunning: boolean;
  materials: Record<string, string>;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <table className="param-table">
        <tbody>
          <tr>
            <td className="param-label">Material</td>
            <td className="param-value">
              <select
                className="form-select"
                value={config.material_name}
                onChange={(e) => onChange({ material_name: e.target.value })}
              >
                {Object.entries(materials).map(([id, name]) => (
                  <option key={id} value={id}>{name}</option>
                ))}
              </select>
            </td>
          </tr>
          <tr>
            <td className="param-label">SF Pressure</td>
            <td className="param-value">
              <input
                style={structInputStyle}
                type="number"
                step="0.1"
                value={config.safety_factor_pressure}
                onChange={(e) => onChange({ safety_factor_pressure: Number(e.target.value) })}
              />
            </td>
          </tr>
          <tr>
            <td className="param-label">SF Thermal</td>
            <td className="param-value">
              <input
                style={structInputStyle}
                type="number"
                step="0.1"
                value={config.safety_factor_thermal}
                onChange={(e) => onChange({ safety_factor_thermal: Number(e.target.value) })}
              />
            </td>
          </tr>
          <tr>
            <td className="param-label">Design Life [cycles]</td>
            <td className="param-value">
              <input
                style={structInputStyle}
                type="number"
                step="10"
                value={config.design_life_cycles}
                onChange={(e) => onChange({ design_life_cycles: Number(e.target.value) })}
              />
            </td>
          </tr>
        </tbody>
      </table>

      <button
        className={`run-btn ${isRunning ? "is-running" : ""}`}
        onClick={onRun}
        disabled={isRunning}
        style={{ marginTop: 4, width: "100%" }}
      >
        <Icon icon={isRunning ? "dot" : "play"} size={12} />
        {isRunning ? "COMPUTING..." : "RUN ANALYSIS"}
      </button>
    </div>
  );
}

function StructuralWorkspace({
  result,
  activeTab,
  onTabChange,
}: {
  result: WallThicknessResponse | null;
  activeTab: string;
  onTabChange: (tab: string) => void;
}) {
  const tabs = ["stress", "safety", "thickness"];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Tab bar */}
      <div className="workspace-tabs">
        {tabs.map((tab) => (
          <button
            key={tab}
            className={`workspace-tab ${activeTab === tab ? "active" : ""}`}
            onClick={() => onTabChange(tab)}
          >
            {tab === "stress"
              ? "Stress Distribution"
              : tab === "safety"
                ? "Safety Factor"
                : "Wall Comparison"}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
        {!result && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              color: "var(--text-muted)",
              fontSize: 12,
            }}
          >
            Select material and run structural analysis
          </div>
        )}

        {result && activeTab === "stress" && (
          <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
            <div style={{ marginBottom: 8, fontWeight: 600 }}>Stress vs Axial Position</div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr>
                  <th style={thStyle}>x [mm]</th>
                  <th style={thStyle}>Hoop [MPa]</th>
                  <th style={thStyle}>Thermal [MPa]</th>
                  <th style={thStyle}>von Mises [MPa]</th>
                </tr>
              </thead>
              <tbody>
                {sampleIndices(result.x_mm.length, 15).map((i) => (
                  <tr key={i}>
                    <td style={tdStyle}>{result.x_mm[i].toFixed(1)}</td>
                    <td style={tdStyle}>{result.hoop_stress_mpa[i].toFixed(1)}</td>
                    <td style={tdStyle}>{result.thermal_stress_mpa[i].toFixed(1)}</td>
                    <td style={tdStyle}>{result.von_mises_mpa[i].toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {result && activeTab === "safety" && (
          <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
            <div style={{ marginBottom: 8, fontWeight: 600 }}>Safety Factor vs Position</div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr>
                  <th style={thStyle}>x [mm]</th>
                  <th style={thStyle}>SF</th>
                </tr>
              </thead>
              <tbody>
                {sampleIndices(result.x_mm.length, 15).map((i) => (
                  <tr key={i}>
                    <td style={tdStyle}>{result.x_mm[i].toFixed(1)}</td>
                    <td
                      style={{
                        ...tdStyle,
                        color: result.safety_factor[i] < 1.0 ? "#e65100" : undefined,
                      }}
                    >
                      {result.safety_factor[i].toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {result && activeTab === "thickness" && (
          <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
            <div style={{ marginBottom: 8, fontWeight: 600 }}>
              Wall Thickness: Required vs Actual [mm]
            </div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr>
                  <th style={thStyle}>x [mm]</th>
                  <th style={thStyle}>Min (Pressure)</th>
                  <th style={thStyle}>Max (Thermal)</th>
                  <th style={thStyle}>Actual</th>
                </tr>
              </thead>
              <tbody>
                {sampleIndices(result.x_mm.length, 15).map((i) => (
                  <tr key={i}>
                    <td style={tdStyle}>{result.x_mm[i].toFixed(1)}</td>
                    <td style={tdStyle}>
                      {result.min_thickness_pressure_mm[i].toFixed(3)}
                    </td>
                    <td style={tdStyle}>
                      {result.min_thickness_thermal_mm[i] > 100
                        ? "\u221E"
                        : result.min_thickness_thermal_mm[i].toFixed(3)}
                    </td>
                    <td style={tdStyle}>
                      {result.actual_thickness_mm[i].toFixed(3)}
                    </td>
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

const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "4px 8px",
  borderBottom: "1px solid var(--border-subtle)",
  color: "var(--text-muted)",
  fontWeight: 600,
};

const tdStyle: React.CSSProperties = {
  padding: "3px 8px",
  fontFamily: "var(--font-mono)",
  borderBottom: "1px solid var(--border-subtle)",
};

function sampleIndices(total: number, count: number): number[] {
  if (total <= count) return Array.from({ length: total }, (_, i) => i);
  const step = (total - 1) / (count - 1);
  return Array.from({ length: count }, (_, i) => Math.round(i * step));
}

function StructuralMetrics({ result }: { result: WallThicknessResponse | null }) {
  if (!result) {
    return (
      <div style={{ padding: "12px 16px", fontSize: 11, color: "var(--text-muted)" }}>
        No results yet
      </div>
    );
  }

  const pass = result.min_safety_factor >= 1.0;

  return (
    <div style={{ padding: "12px 16px" }}>
      <div
        style={{
          fontSize: 10,
          fontWeight: 600,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          color: "var(--text-muted)",
          marginBottom: 12,
        }}
      >
        Structural Metrics
      </div>

      {/* Pass/fail indicator */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          padding: "6px 10px",
          borderRadius: 4,
          background: pass ? "#1b3a1b" : "#3a1b1b",
          marginBottom: 12,
        }}
      >
        <Icon icon={pass ? "tick-circle" : "cross-circle"} size={14} style={{ color: pass ? "#43a047" : "#e53935" }} />
        <span style={{ fontSize: 11, fontWeight: 600, color: pass ? "#43a047" : "#e53935" }}>
          {pass ? "PASS" : "FAIL"}
        </span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <MetricRow
          label="Min Safety Factor"
          value={result.min_safety_factor.toFixed(2)}
          warn={result.min_safety_factor < 1.5}
        />
        <MetricRow
          label="Critical Station"
          value={`${result.critical_station_x_mm.toFixed(1)} mm`}
        />
        <MetricRow
          label="Max Hoop Stress"
          value={`${result.max_hoop_stress_mpa.toFixed(1)} MPa`}
        />
        <MetricRow
          label="Max Thermal Stress"
          value={`${result.max_thermal_stress_mpa.toFixed(1)} MPa`}
        />
        <MetricRow
          label="Max von Mises"
          value={`${result.max_von_mises_mpa.toFixed(1)} MPa`}
        />
      </div>

      {/* Material info */}
      <div
        style={{
          marginTop: 16,
          fontSize: 10,
          fontWeight: 600,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          color: "var(--text-muted)",
          marginBottom: 8,
        }}
      >
        Material
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <MetricRow label="Name" value={result.material.name} />
        <MetricRow
          label="Yield"
          value={`${result.material.yield_strength_mpa.toFixed(0)} MPa`}
        />
        <MetricRow
          label="Conductivity"
          value={`${result.material.thermal_conductivity_w_mk.toFixed(1)} W/mK`}
        />
        <MetricRow
          label="Max Temp"
          value={`${result.material.max_service_temp_k.toFixed(0)} K`}
        />
      </div>

      {result.warnings.length > 0 && (
        <div style={{ marginTop: 12 }}>
          {result.warnings.map((w, i) => (
            <div
              key={i}
              style={{
                fontSize: 10,
                color: "#e65100",
                display: "flex",
                gap: 4,
                alignItems: "flex-start",
                marginBottom: 4,
              }}
            >
              <Icon icon="warning-sign" size={10} />
              <span>{w}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function MetricRow({
  label,
  value,
  warn,
}: {
  label: string;
  value: string;
  warn?: boolean;
}) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
      <span style={{ color: "var(--text-muted)" }}>{label}</span>
      <span
        style={{
          fontFamily: "var(--font-mono)",
          color: warn ? "#e65100" : "var(--text-primary)",
        }}
      >
        {value}
      </span>
    </div>
  );
}

export default function StructuralPage() {
  const {
    sessionId,
    wallThicknessConfig,
    wallThicknessResult,
    moduleStatus,
    setWallThicknessConfig,
    setWallThicknessResult,
    markModuleCompleted,
  } = useDesignSessionStore();

  const [activeTab, setActiveTab] = useState("stress");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [materials, setMaterials] = useState<Record<string, string>>({
    inconel718: "Inconel 718",
    inconel625: "Inconel 625",
    copper_c18150: "Copper C18150",
    stainless316l: "Stainless 316L",
    haynes230: "Haynes 230",
  });

  const mutation = useWallThicknessMutation();
  const isRunning = mutation.isPending;

  useEffect(() => {
    listMaterials()
      .then((res) => setMaterials(res.materials))
      .catch(() => {
        /* use defaults */
      });
  }, []);

  const handleRun = useCallback(async () => {
    if (!sessionId) return;
    setErrorMsg(null);

    try {
      const result = await mutation.mutateAsync({
        sessionId,
        config: wallThicknessConfig,
      });
      setWallThicknessResult(result);
      markModuleCompleted("wall_thickness");
      setActiveTab("safety");
    } catch (err) {
      setErrorMsg(extractError(err));
    }
  }, [sessionId, wallThicknessConfig, mutation, setWallThicknessResult, markModuleCompleted]);

  return (
    <ModuleGate requires={["engine", "cooling"]}>
      {/* Top bar */}
      <header className="app-topbar">
        <div className="topbar-logo">
          <div className="topbar-logo-dot" />
          RESA
        </div>
        <div className="topbar-breadcrumb">
          <span className="page-name">Wall Thickness</span>
        </div>
        <div className="topbar-spacer" />
        <div className="topbar-actions">
          <button
            className={`run-btn ${isRunning ? "is-running" : ""}`}
            onClick={handleRun}
            disabled={isRunning}
          >
            <Icon icon={isRunning ? "dot" : "play"} size={12} />
            {isRunning ? "COMPUTING..." : "RUN ANALYSIS"}
          </button>
        </div>
      </header>

      {/* Left panel */}
      <div className="app-left-panel">
        <StaleDataBanner moduleName="wall_thickness" moduleStatus={moduleStatus} />
        <div
          style={{
            flexShrink: 0,
            padding: "7px 12px",
            borderBottom: "1px solid var(--border-subtle)",
          }}
        >
          <span
            style={{
              fontSize: 10,
              fontWeight: 600,
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: "var(--text-muted)",
            }}
          >
            Structural Parameters
          </span>
        </div>
        <div className="panel-scroll" style={{ padding: "12px 16px" }}>
          <StructuralConfigForm
            config={wallThicknessConfig}
            onChange={setWallThicknessConfig}
            onRun={handleRun}
            isRunning={isRunning}
            materials={materials}
          />
        </div>
      </div>

      {/* Center workspace */}
      <div className="app-workspace">
        <StructuralWorkspace
          result={wallThicknessResult}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />
      </div>

      {/* Right panel */}
      <div className="app-right-panel">
        <StructuralMetrics result={wallThicknessResult} />
      </div>

      {/* Status bar */}
      <footer className="app-statusbar">
        <div className="statusbar-item">
          <Icon
            icon={isRunning ? "dot" : errorMsg ? "cross" : wallThicknessResult ? "tick-circle" : "circle"}
            size={10}
          />
          <span>
            {isRunning
              ? "COMPUTING"
              : errorMsg
                ? "ERROR"
                : wallThicknessResult
                  ? "NOMINAL"
                  : "READY"}
          </span>
        </div>
        {errorMsg && (
          <div className="statusbar-item" style={{ color: "var(--red)" }}>
            {errorMsg.length > 80 ? errorMsg.slice(0, 80) + "\u2026" : errorMsg}
          </div>
        )}
        <div className="statusbar-spacer" />
        <div className="statusbar-item">
          <Icon icon="flame" size={10} />
          RESA v2.0.0
        </div>
      </footer>
    </ModuleGate>
  );
}
