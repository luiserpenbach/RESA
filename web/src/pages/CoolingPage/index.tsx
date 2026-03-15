import { useState, useCallback } from "react";
import { Icon } from "@blueprintjs/core";
import { ModuleGate } from "../../components/common/ModuleGate";
import { StaleDataBanner } from "../../components/common/StaleDataBanner";
import { useDesignSessionStore } from "../../store/designSessionStore";
import { useDesignChannelsMutation, useAnalyzeCoolingMutation } from "../../api/cooling";
import type { CoolingChannelConfig, CoolingChannelResponse, CoolingAnalysisResponse } from "../../types/cooling";

function extractError(err: unknown): string {
  if (err && typeof err === "object" && "response" in err) {
    const resp = (err as { response?: { data?: { detail?: unknown } } }).response;
    const detail = resp?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  if (err instanceof Error) return err.message;
  return "Cooling analysis failed.";
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
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Channel type */}
      <div>
        <label className="form-label">Channel Type</label>
        <select
          className="form-select"
          value={config.channel_type}
          onChange={(e) => onChange({ channel_type: e.target.value as "rectangular" | "trapezoidal" })}
        >
          <option value="rectangular">Rectangular</option>
          <option value="trapezoidal">Trapezoidal</option>
        </select>
      </div>

      {/* Height profile */}
      <div>
        <label className="form-label">Height Profile</label>
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
      </div>

      {/* Heights */}
      <div>
        <label className="form-label">Height at Throat [mm]</label>
        <input
          className="form-input"
          type="number"
          step="0.1"
          value={config.height_throat_m * 1e3}
          onChange={(e) => onChange({ height_throat_m: Number(e.target.value) / 1e3 })}
        />
      </div>

      {config.height_profile === "tapered" && (
        <>
          <div>
            <label className="form-label">Height at Chamber [mm]</label>
            <input
              className="form-input"
              type="number"
              step="0.1"
              value={config.height_chamber_m * 1e3}
              onChange={(e) => onChange({ height_chamber_m: Number(e.target.value) / 1e3 })}
            />
          </div>
          <div>
            <label className="form-label">Height at Exit [mm]</label>
            <input
              className="form-input"
              type="number"
              step="0.1"
              value={config.height_exit_m * 1e3}
              onChange={(e) => onChange({ height_exit_m: Number(e.target.value) / 1e3 })}
            />
          </div>
        </>
      )}

      {config.channel_type === "trapezoidal" && (
        <div>
          <label className="form-label">Taper Angle [deg]</label>
          <input
            className="form-input"
            type="number"
            step="1"
            value={config.taper_angle_deg}
            onChange={(e) => onChange({ taper_angle_deg: Number(e.target.value) })}
          />
        </div>
      )}

      {/* Channels override */}
      <div>
        <label className="form-label">Channel Count Override</label>
        <input
          className="form-input"
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
      </div>

      {/* Aspect ratio limit */}
      <div>
        <label className="form-label">Max Aspect Ratio</label>
        <input
          className="form-input"
          type="number"
          step="1"
          value={config.aspect_ratio_limit}
          onChange={(e) => onChange({ aspect_ratio_limit: Number(e.target.value) })}
        />
      </div>

      {/* Run button */}
      <button
        className={`run-btn ${isRunning ? "is-running" : ""}`}
        onClick={onRun}
        disabled={isRunning}
        style={{ marginTop: 12, width: "100%" }}
      >
        <Icon icon={isRunning ? "dot" : "play"} size={12} />
        {isRunning ? "COMPUTING..." : "RUN COOLING"}
      </button>
    </div>
  );
}

function CoolingWorkspace({
  channelResult,
  analysisResult,
  activeTab,
  onTabChange,
}: {
  channelResult: CoolingChannelResponse | null;
  analysisResult: CoolingAnalysisResponse | null;
  activeTab: string;
  onTabChange: (tab: string) => void;
}) {
  const tabs = ["channels", "thermal"];

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
            {tab === "channels" ? "Channel Profile" : "Thermal Dashboard"}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
        {activeTab === "channels" && channelResult && (
          <div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>
              {channelResult.num_channels} channels | {channelResult.channel_type} |{" "}
              {channelResult.height_profile} profile
            </div>
            <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
              Width range: {channelResult.min_channel_width_mm.toFixed(2)} -{" "}
              {channelResult.max_channel_width_mm.toFixed(2)} mm
            </div>
            <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
              Aspect ratio: {channelResult.min_aspect_ratio.toFixed(1)} -{" "}
              {channelResult.max_aspect_ratio.toFixed(1)}
            </div>
          </div>
        )}

        {activeTab === "thermal" && analysisResult && analysisResult.figure_thermal && (
          <div
            style={{ width: "100%", height: "100%" }}
            ref={(el) => {
              if (el && analysisResult.figure_thermal) {
                try {
                  const Plotly = (window as unknown as { Plotly?: { newPlot: Function } }).Plotly;
                  if (Plotly) {
                    const fig = JSON.parse(analysisResult.figure_thermal);
                    Plotly.newPlot(el, fig.data, {
                      ...fig.layout,
                      autosize: true,
                      paper_bgcolor: "transparent",
                      plot_bgcolor: "transparent",
                    });
                  }
                } catch {
                  /* ignore parse errors */
                }
              }
            }}
          />
        )}

        {activeTab === "thermal" && analysisResult && !analysisResult.figure_thermal && (
          <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
            Thermal results available. Max wall temp: {analysisResult.max_wall_temp_k.toFixed(0)} K
          </div>
        )}

        {!channelResult && !analysisResult && (
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
            label="Max AR"
            value={channelResult.max_aspect_ratio.toFixed(1)}
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
        <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
          No results yet
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
  }, [sessionId, coolingConfig, channelsMutation, analysisMutation, setCoolingChannelResult, setCoolingAnalysisResult, markModuleCompleted]);

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
            icon={isRunning ? "dot" : errorMsg ? "cross" : coolingAnalysisResult ? "tick-circle" : "circle"}
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
