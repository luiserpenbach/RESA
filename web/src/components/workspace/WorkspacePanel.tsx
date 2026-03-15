import { Icon } from "@blueprintjs/core";
import { PlotlyRenderer } from "../plots/PlotlyRenderer";
import { ParameterStudyPanel } from "./ParameterStudyPanel";
import { SuggestedValuesPanel } from "./SuggestedValuesPanel";
import { useUiStore, type WorkspaceTab } from "../../store/uiStore";
import type { EngineDesignResponse, EngineConfigRequest } from "../../types/engine";

interface WorkspacePanelProps {
  config: EngineConfigRequest;
  result: EngineDesignResponse | null;
  isLoading: boolean;
}

const TABS: { id: WorkspaceTab; label: string; icon: string }[] = [
  { id: "dashboard",        label: "Dashboard",        icon: "timeline-line-chart" },
  { id: "parameter_study",  label: "Parameter Study",  icon: "regression-chart" },
  { id: "suggested_values", label: "Suggested Values", icon: "book" },
];

export function WorkspacePanel({ config, result, isLoading }: WorkspacePanelProps) {
  const { workspaceTab, setWorkspaceTab } = useUiStore();

  return (
    <>
      {/* Tab bar */}
      <div className="workspace-tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`workspace-tab ${workspaceTab === t.id ? "active" : ""}`}
            onClick={() => setWorkspaceTab(t.id)}
          >
            <Icon icon={t.icon as never} size={11} />
            {t.label}
          </button>
        ))}

        <div style={{ flex: 1 }} />

        {result && (
          <div style={{
            display: "flex", alignItems: "center", gap: 6,
            padding: "0 14px", fontSize: 10,
            color: "var(--text-muted)",
            fontFamily: "var(--font-mono)",
          }}>
            <span style={{ color: "var(--green)" }}>●</span>
            {config.fuel} / {config.oxidizer} · Pc {result.combustion?.pc_bar?.toFixed(1)} bar
          </div>
        )}
      </div>

      {/* Content */}
      <div className="workspace-content">
        {isLoading && <LoadingPane />}

        {/* Dashboard: primary 4-panel figure (contour + Mach + T + P) */}
        {!isLoading && workspaceTab === "dashboard" && result?.figure_dashboard && (
          <div style={{ padding: 16 }}>
            <PlotlyRenderer figureJson={result.figure_dashboard} height={640} />
          </div>
        )}

        {!isLoading && workspaceTab === "dashboard" && !result && (
          <EmptyPane message="Run a design to see the dashboard." />
        )}

        {!isLoading && workspaceTab === "dashboard" && result && !result.figure_dashboard && (
          <EmptyPane message="Dashboard figure not available. Check API logs." />
        )}

        {/* Parameter Study: CEA sweeps */}
        {!isLoading && workspaceTab === "parameter_study" && (
          <ParameterStudyPanel config={config} />
        )}

        {/* Suggested Values: empirical reference charts */}
        {!isLoading && workspaceTab === "suggested_values" && (
          <SuggestedValuesPanel config={config} result={result} />
        )}
      </div>
    </>
  );
}

function LoadingPane() {
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      height: "100%",
      gap: 20,
    }}>
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 4,
        padding: 32,
      }}>
        {[0.4, 0.7, 1.0, 0.85, 0.6, 0.45, 0.35].map((h, i) => (
          <div
            key={i}
            style={{
              width: 6,
              height: 40 * h,
              background: "var(--accent)",
              borderRadius: 2,
              animation: `bar-pulse 1.2s ease-in-out ${i * 0.1}s infinite`,
              opacity: 0.7,
            }}
          />
        ))}
      </div>
      <div style={{
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        color: "var(--text-secondary)",
        letterSpacing: "0.1em",
      }}>
        SOLVING ENGINE DESIGN…
      </div>
      <style>{`
        @keyframes bar-pulse {
          0%, 100% { transform: scaleY(0.4); opacity: 0.4; }
          50%       { transform: scaleY(1.0); opacity: 0.9; }
        }
      `}</style>
    </div>
  );
}

function EmptyPane({ message }: { message: string }) {
  return (
    <div className="empty-state">
      <div className="empty-state-icon">
        <Icon icon="play" size={36} />
      </div>
      <div className="empty-state-text">{message}</div>
    </div>
  );
}
