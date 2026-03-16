import React, { useState, useCallback, useMemo, useEffect } from "react";
import { Icon } from "@blueprintjs/core";
import { ModuleGate } from "../../components/common/ModuleGate";
import { StaleDataBanner } from "../../components/common/StaleDataBanner";
import { PlotlyRenderer } from "../../components/plots/PlotlyRenderer";
import { useDesignSessionStore } from "../../store/designSessionStore";
import {
  useDesignChannelsMutation,
  useAnalyzeCoolingMutation,
  useCrossSectionQuery,
} from "../../api/cooling";
import type {
  CoolingChannelConfig,
  CoolingChannelResponse,
  CoolingAnalysisResponse,
} from "../../types/cooling";

function extractError(err: unknown): string {
  if (err && typeof err === "object" && "response" in err) {
    const resp = (err as { response?: { data?: { detail?: unknown } } }).response;
    const detail = resp?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  if (err instanceof Error) return err.message;
  return "Cooling analysis failed.";
}

const coolingInputStyle: React.CSSProperties = {
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

const sectionLabelStyle: React.CSSProperties = {
  fontSize: 10,
  fontWeight: 600,
  letterSpacing: "0.10em",
  textTransform: "uppercase" as const,
  color: "var(--text-muted)",
  marginBottom: 6,
  marginTop: 10,
};

/** Nullable number input: empty string maps to null, number maps to number */
function NullableNumberInput({
  value,
  onChange,
  placeholder,
  step,
  min,
}: {
  value: number | null;
  onChange: (v: number | null) => void;
  placeholder?: string;
  step?: string;
  min?: string;
}) {
  return (
    <input
      style={coolingInputStyle}
      type="number"
      step={step ?? "any"}
      min={min}
      placeholder={placeholder ?? "auto"}
      value={value === null ? "" : value}
      onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
    />
  );
}

function CoolingConfigForm({
  config,
  onChange,
  onRun,
  isRunning,
}: {
  config: CoolingChannelConfig;
  onChange: (partial: Partial<CoolingChannelConfig>) => void;
  onRun: () => void;
  isRunning: boolean;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {/* ── Channel Geometry ── */}
      <div style={sectionLabelStyle}>Channel Geometry</div>
      <table className="param-table">
        <tbody>
          <tr>
            <td className="param-label">Channel Type</td>
            <td className="param-value">
              <select
                className="form-select"
                value={config.channel_type}
                onChange={(e) =>
                  onChange({ channel_type: e.target.value as "rectangular" | "trapezoidal" })
                }
              >
                <option value="rectangular">Rectangular</option>
                <option value="trapezoidal">Trapezoidal</option>
              </select>
            </td>
          </tr>
          <tr>
            <td className="param-label">Height Profile</td>
            <td className="param-value">
              <select
                className="form-select"
                value={config.height_profile}
                onChange={(e) =>
                  onChange({ height_profile: e.target.value as "constant" | "tapered" | "custom" })
                }
              >
                <option value="constant">Constant</option>
                <option value="tapered">Tapered</option>
              </select>
            </td>
          </tr>
          <tr>
            <td className="param-label">Height at Throat [mm]</td>
            <td className="param-value">
              <input
                style={coolingInputStyle}
                type="number"
                step="0.1"
                value={config.height_throat_m * 1e3}
                onChange={(e) => onChange({ height_throat_m: Number(e.target.value) / 1e3 })}
              />
            </td>
          </tr>

          {config.height_profile === "tapered" && (
            <>
              <tr>
                <td className="param-label">Height at Chamber [mm]</td>
                <td className="param-value">
                  <input
                    style={coolingInputStyle}
                    type="number"
                    step="0.1"
                    value={config.height_chamber_m * 1e3}
                    onChange={(e) =>
                      onChange({ height_chamber_m: Number(e.target.value) / 1e3 })
                    }
                  />
                </td>
              </tr>
              <tr>
                <td className="param-label">Height at Exit [mm]</td>
                <td className="param-value">
                  <input
                    style={coolingInputStyle}
                    type="number"
                    step="0.1"
                    value={config.height_exit_m * 1e3}
                    onChange={(e) =>
                      onChange({ height_exit_m: Number(e.target.value) / 1e3 })
                    }
                  />
                </td>
              </tr>
            </>
          )}

          {config.channel_type === "trapezoidal" && (
            <tr>
              <td className="param-label">Taper Angle [deg]</td>
              <td className="param-value">
                <input
                  style={coolingInputStyle}
                  type="number"
                  step="1"
                  value={config.taper_angle_deg}
                  onChange={(e) => onChange({ taper_angle_deg: Number(e.target.value) })}
                />
              </td>
            </tr>
          )}

          <tr>
            <td className="param-label">Wall Thickness [mm]</td>
            <td className="param-value">
              <NullableNumberInput
                value={config.wall_thickness_mm}
                onChange={(v) => onChange({ wall_thickness_mm: v })}
                placeholder="from engine"
                step="0.1"
                min="0.1"
              />
            </td>
          </tr>
          <tr>
            <td className="param-label">Rib Width [mm]</td>
            <td className="param-value">
              <NullableNumberInput
                value={config.rib_width_throat_mm}
                onChange={(v) => onChange({ rib_width_throat_mm: v })}
                placeholder="from engine"
                step="0.1"
                min="0.1"
              />
            </td>
          </tr>
          <tr>
            <td className="param-label">Helix Angle [deg]</td>
            <td className="param-value">
              <input
                style={coolingInputStyle}
                type="number"
                step="1"
                min="0"
                max="45"
                value={config.helix_angle_deg}
                onChange={(e) => onChange({ helix_angle_deg: Number(e.target.value) })}
              />
            </td>
          </tr>
          <tr>
            <td className="param-label">Wall Roughness [µm]</td>
            <td className="param-value">
              <NullableNumberInput
                value={config.roughness_microns}
                onChange={(v) => onChange({ roughness_microns: v })}
                placeholder="from engine"
                step="1"
                min="0"
              />
            </td>
          </tr>
          <tr>
            <td className="param-label">Channel Count Override</td>
            <td className="param-value">
              <input
                style={coolingInputStyle}
                type="number"
                step="1"
                placeholder="Auto"
                value={config.num_channels_override ?? ""}
                onChange={(e) =>
                  onChange({
                    num_channels_override: e.target.value ? Number(e.target.value) : null,
                  })
                }
              />
            </td>
          </tr>
          <tr>
            <td className="param-label">Max Aspect Ratio</td>
            <td className="param-value">
              <input
                style={coolingInputStyle}
                type="number"
                step="1"
                value={config.aspect_ratio_limit}
                onChange={(e) => onChange({ aspect_ratio_limit: Number(e.target.value) })}
              />
            </td>
          </tr>
        </tbody>
      </table>

      {/* ── Axial Margins ── */}
      <div style={sectionLabelStyle}>Axial Margins (CAD)</div>
      <table className="param-table">
        <tbody>
          <tr>
            <td className="param-label">Start Margin [mm]</td>
            <td className="param-value">
              <input
                style={coolingInputStyle}
                type="number"
                step="1"
                min="0"
                value={config.start_margin_mm}
                onChange={(e) => onChange({ start_margin_mm: Number(e.target.value) })}
              />
            </td>
          </tr>
          <tr>
            <td className="param-label">End Margin [mm]</td>
            <td className="param-value">
              <input
                style={coolingInputStyle}
                type="number"
                step="1"
                min="0"
                value={config.end_margin_mm}
                onChange={(e) => onChange({ end_margin_mm: Number(e.target.value) })}
              />
            </td>
          </tr>
        </tbody>
      </table>

      {/* ── Coolant Conditions ── */}
      <div style={sectionLabelStyle}>Coolant Conditions</div>
      <table className="param-table">
        <tbody>
          <tr>
            <td className="param-label">Inlet Pressure [bar]</td>
            <td className="param-value">
              <NullableNumberInput
                value={config.coolant_p_in_bar}
                onChange={(v) => onChange({ coolant_p_in_bar: v })}
                placeholder="from engine"
                step="1"
                min="0"
              />
            </td>
          </tr>
          <tr>
            <td className="param-label">Inlet Temperature [K]</td>
            <td className="param-value">
              <NullableNumberInput
                value={config.coolant_t_in_k}
                onChange={(v) => onChange({ coolant_t_in_k: v })}
                placeholder="from engine"
                step="1"
                min="0"
              />
            </td>
          </tr>
          <tr>
            <td className="param-label">Mass Fraction</td>
            <td className="param-value">
              <NullableNumberInput
                value={config.coolant_mass_fraction}
                onChange={(v) => onChange({ coolant_mass_fraction: v })}
                placeholder="from engine"
                step="0.05"
                min="0"
              />
            </td>
          </tr>
        </tbody>
      </table>

      <button
        className={`run-btn ${isRunning ? "is-running" : ""}`}
        onClick={onRun}
        disabled={isRunning}
        style={{ marginTop: 8, width: "100%" }}
      >
        <Icon icon={isRunning ? "dot" : "play"} size={12} />
        {isRunning ? "COMPUTING..." : "RUN COOLING"}
      </button>
    </div>
  );
}

function CrossSectionPanel({
  sessionId,
  channelResult,
}: {
  sessionId: string | null;
  channelResult: CoolingChannelResponse | null;
}) {
  const n = channelResult ? channelResult.x_mm.length : 0;
  const midpoint = n > 0 ? Math.floor(n / 2) : 0;
  const [stationIdx, setStationIdx] = useState(midpoint);
  const [debouncedIdx, setDebouncedIdx] = useState(midpoint);

  // Reset to midpoint when channel result changes
  useEffect(() => {
    const mid = n > 0 ? Math.floor(n / 2) : 0;
    setStationIdx(mid);
    setDebouncedIdx(mid);
  }, [n]);

  // Debounce slider
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedIdx(stationIdx), 200);
    return () => clearTimeout(timer);
  }, [stationIdx]);

  const { data, isFetching } = useCrossSectionQuery(
    sessionId,
    debouncedIdx,
    !!channelResult
  );

  if (!channelResult) {
    return (
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
        Run cooling to enable cross-section view
      </div>
    );
  }

  const currentX = channelResult.x_mm[stationIdx]?.toFixed(1) ?? "—";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {/* Slider controls */}
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 11, color: "var(--text-muted)", whiteSpace: "nowrap" }}>
          Axial position:
        </span>
        <input
          type="range"
          min={0}
          max={Math.max(0, n - 1)}
          step={1}
          value={stationIdx}
          onChange={(e) => setStationIdx(Number(e.target.value))}
          style={{ flex: 1 }}
        />
        <span
          style={{
            fontSize: 11,
            fontFamily: "var(--font-mono)",
            color: "var(--text-primary)",
            minWidth: 70,
            textAlign: "right",
          }}
        >
          {currentX} mm
        </span>
        {isFetching && (
          <Icon icon="dot" size={10} style={{ color: "var(--text-muted)" }} />
        )}
      </div>
      <PlotlyRenderer figureJson={data?.figure ?? null} height={440} />
    </div>
  );
}

function CoolingWorkspace({
  sessionId,
  channelResult,
  analysisResult,
  activeTab,
  onTabChange,
}: {
  sessionId: string | null;
  channelResult: CoolingChannelResponse | null;
  analysisResult: CoolingAnalysisResponse | null;
  activeTab: string;
  onTabChange: (tab: string) => void;
}) {
  const baseTabsAlways = ["channels", "cross-section", "3d-view", "thermal"];
  const n2oTabs =
    analysisResult?.is_n2o_analysis
      ? ["t-rho", "p-t"]
      : [];
  const tabs = [...baseTabsAlways, ...n2oTabs];

  // Build a Plotly figure JSON for the channel profile from the response arrays.
  const channelFigureJson = useMemo(() => {
    if (!channelResult || !channelResult.x_mm.length) return null;
    return JSON.stringify({
      data: [
        {
          type: "scatter",
          x: channelResult.x_mm,
          y: channelResult.channel_width_mm,
          name: "Channel Width",
          mode: "lines",
          line: { color: "#4a9eff", width: 2 },
          hovertemplate: "X: %{x:.1f} mm<br>Width: %{y:.3f} mm<extra></extra>",
        },
        {
          type: "scatter",
          x: channelResult.x_mm,
          y: channelResult.channel_height_mm,
          name: "Channel Height",
          mode: "lines",
          line: { color: "#2ecc71", width: 2 },
          hovertemplate: "X: %{x:.1f} mm<br>Height: %{y:.3f} mm<extra></extra>",
        },
        {
          type: "scatter",
          x: channelResult.x_mm,
          y: channelResult.rib_width_mm,
          name: "Rib Width",
          mode: "lines",
          line: { color: "#f39c12", width: 2, dash: "dot" },
          hovertemplate: "X: %{x:.1f} mm<br>Rib: %{y:.3f} mm<extra></extra>",
        },
      ],
      layout: {
        xaxis: {
          title: "Axial Position [mm]",
          color: "#a0a0a0",
          gridcolor: "#252525",
          linecolor: "#2e2e2e",
        },
        yaxis: {
          title: "Dimension [mm]",
          color: "#a0a0a0",
          gridcolor: "#252525",
          linecolor: "#2e2e2e",
        },
        legend: {
          font: { color: "#a0a0a0" },
          bgcolor: "rgba(22,22,22,0.85)",
          bordercolor: "#2e2e2e",
          borderwidth: 1,
        },
        paper_bgcolor: "#161616",
        plot_bgcolor: "#111111",
        font: { color: "#e2e2e2", size: 11 },
        margin: { l: 50, r: 20, t: 30, b: 50 },
      },
    });
  }, [channelResult]);

  const tabLabel = (tab: string) => {
    switch (tab) {
      case "channels":
        return "Channel Profile";
      case "cross-section":
        return "Cross-Section";
      case "3d-view":
        return "3D View";
      case "thermal":
        return "Thermal Dashboard";
      case "t-rho":
        return "T-ρ Diagram";
      case "p-t":
        return "P-T Diagram";
      default:
        return tab;
    }
  };

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
            {tabLabel(tab)}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
        {activeTab === "channels" && channelResult && (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
              {channelResult.num_channels} channels | {channelResult.channel_type} |{" "}
              {channelResult.height_profile} profile
            </div>
            <PlotlyRenderer figureJson={channelFigureJson} height={420} />
          </div>
        )}

        {activeTab === "cross-section" && (
          <CrossSectionPanel sessionId={sessionId} channelResult={channelResult} />
        )}

        {activeTab === "3d-view" && channelResult && (
          <PlotlyRenderer figureJson={channelResult.figure_3d} height={500} />
        )}

        {activeTab === "thermal" && analysisResult && (
          <PlotlyRenderer figureJson={analysisResult.figure_thermal} height={500} />
        )}

        {activeTab === "t-rho" && analysisResult && (
          <PlotlyRenderer figureJson={analysisResult.figure_t_rho} height={500} />
        )}

        {activeTab === "p-t" && analysisResult && (
          <PlotlyRenderer figureJson={analysisResult.figure_p_t} height={500} />
        )}

        {!channelResult && !analysisResult && activeTab !== "cross-section" && (
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
            Configure cooling channels and run analysis
          </div>
        )}
      </div>
    </div>
  );
}

function CoolingMetrics({
  channelResult,
  analysisResult,
}: {
  channelResult: CoolingChannelResponse | null;
  analysisResult: CoolingAnalysisResponse | null;
}) {
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
        Cooling Metrics
      </div>

      {channelResult && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <MetricRow label="Channels" value={String(channelResult.num_channels)} />
          <MetricRow
            label="Min Width"
            value={`${channelResult.min_channel_width_mm.toFixed(2)} mm`}
          />
          <MetricRow
            label="Max Width"
            value={`${channelResult.max_channel_width_mm.toFixed(2)} mm`}
          />
          <MetricRow
            label="Max AR"
            value={channelResult.max_aspect_ratio.toFixed(1)}
          />
          <MetricRow
            label="Wall Thickness"
            value={`${channelResult.wall_thickness_mm_val.toFixed(2)} mm`}
          />
          <MetricRow
            label="Axial Span"
            value={
              channelResult.x_mm.length > 1
                ? `${(
                    channelResult.x_mm[channelResult.x_mm.length - 1] -
                    channelResult.x_mm[0]
                  ).toFixed(1)} mm`
                : "—"
            }
          />
        </div>
      )}

      {analysisResult && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 16 }}>
          <div
            style={{
              fontSize: 10,
              fontWeight: 600,
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: "var(--text-muted)",
              marginBottom: 4,
            }}
          >
            Thermal
          </div>
          <MetricRow
            label="Max Wall Temp"
            value={`${analysisResult.max_wall_temp_k.toFixed(0)} K`}
            warn={analysisResult.max_wall_temp_k > 900}
          />
          <MetricRow
            label="Max Heat Flux"
            value={`${analysisResult.max_heat_flux_mw_m2.toFixed(2)} MW/m\u00B2`}
          />
          <MetricRow
            label="Pressure Drop"
            value={`${analysisResult.pressure_drop_bar.toFixed(2)} bar`}
          />
          <MetricRow
            label="Outlet Temp"
            value={`${analysisResult.outlet_temp_k.toFixed(1)} K`}
          />

          {analysisResult.is_n2o_analysis && (
            <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 12 }}>
              <div
                style={{
                  fontSize: 10,
                  fontWeight: 600,
                  letterSpacing: "0.12em",
                  textTransform: "uppercase",
                  color: "var(--text-muted)",
                  marginBottom: 4,
                }}
              >
                N₂O Two-Phase
              </div>
              {analysisResult.min_chf_margin != null && (
                <MetricRow
                  label="Min CHF Margin"
                  value={analysisResult.min_chf_margin.toFixed(3)}
                  warn={analysisResult.min_chf_margin > 0.5}
                />
              )}
              {analysisResult.max_quality != null && (
                <MetricRow
                  label="Max Quality"
                  value={analysisResult.max_quality.toFixed(4)}
                  warn={analysisResult.max_quality > 0.3}
                />
              )}
            </div>
          )}

          {analysisResult.warnings.length > 0 && (
            <div style={{ marginTop: 8 }}>
              {analysisResult.warnings.map((w, i) => (
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
      )}

      {!channelResult && !analysisResult && (
        <div style={{ fontSize: 11, color: "var(--text-muted)" }}>No results yet</div>
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
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        fontSize: 11,
      }}
    >
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

export default function CoolingPage() {
  const {
    sessionId,
    coolingConfig,
    coolingChannelResult,
    coolingAnalysisResult,
    moduleStatus,
    setCoolingConfig,
    setCoolingChannelResult,
    setCoolingAnalysisResult,
    markModuleCompleted,
  } = useDesignSessionStore();

  const [activeTab, setActiveTab] = useState("channels");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const channelsMutation = useDesignChannelsMutation();
  const analysisMutation = useAnalyzeCoolingMutation();

  const isRunning = channelsMutation.isPending || analysisMutation.isPending;

  const handleRun = useCallback(async () => {
    if (!sessionId) return;
    setErrorMsg(null);

    try {
      // Step 1: Design channels
      const channels = await channelsMutation.mutateAsync({
        sessionId,
        config: coolingConfig,
      });
      setCoolingChannelResult(channels);
      markModuleCompleted("cooling_channels");

      // Step 2: Run thermal analysis
      const analysis = await analysisMutation.mutateAsync(sessionId);
      setCoolingAnalysisResult(analysis);
      markModuleCompleted("cooling");
      setActiveTab("thermal");
    } catch (err) {
      setErrorMsg(extractError(err));
    }
  }, [
    sessionId,
    coolingConfig,
    channelsMutation,
    analysisMutation,
    setCoolingChannelResult,
    setCoolingAnalysisResult,
    markModuleCompleted,
  ]);

  return (
    <ModuleGate requires={["engine"]}>
      {/* Top bar */}
      <header className="app-topbar">
        <div className="topbar-logo">
          <div className="topbar-logo-dot" />
          RESA
        </div>
        <div className="topbar-breadcrumb">
          <span className="page-name">Cooling Design</span>
        </div>
        <div className="topbar-spacer" />
        <div className="topbar-actions">
          <button
            className={`run-btn ${isRunning ? "is-running" : ""}`}
            onClick={handleRun}
            disabled={isRunning}
          >
            <Icon icon={isRunning ? "dot" : "play"} size={12} />
            {isRunning ? "COMPUTING..." : "RUN COOLING"}
          </button>
        </div>
      </header>

      {/* Left panel */}
      <div className="app-left-panel">
        <StaleDataBanner moduleName="cooling" moduleStatus={moduleStatus} />
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
            Channel Parameters
          </span>
        </div>
        <div className="panel-scroll" style={{ padding: "12px 16px" }}>
          <CoolingConfigForm
            config={coolingConfig}
            onChange={setCoolingConfig}
            onRun={handleRun}
            isRunning={isRunning}
          />
        </div>
      </div>

      {/* Center workspace */}
      <div className="app-workspace">
        <CoolingWorkspace
          sessionId={sessionId}
          channelResult={coolingChannelResult}
          analysisResult={coolingAnalysisResult}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />
      </div>

      {/* Right panel */}
      <div className="app-right-panel">
        <CoolingMetrics
          channelResult={coolingChannelResult}
          analysisResult={coolingAnalysisResult}
        />
      </div>

      {/* Status bar */}
      <footer className="app-statusbar">
        <div className="statusbar-item">
          <Icon
            icon={
              isRunning
                ? "dot"
                : errorMsg
                ? "cross"
                : coolingAnalysisResult
                ? "tick-circle"
                : "circle"
            }
            size={10}
          />
          <span>
            {isRunning
              ? "COMPUTING"
              : errorMsg
              ? "ERROR"
              : coolingAnalysisResult
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
