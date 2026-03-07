import { Icon } from "@blueprintjs/core";
import { SchematicView } from "./SchematicView";
import { PlotlyRenderer } from "../plots/PlotlyRenderer";
import { useUiStore, type WorkspaceTab } from "../../store/uiStore";
import type { EngineDesignResponse, EngineConfigRequest } from "../../types/engine";

interface WorkspacePanelProps {
  config: EngineConfigRequest;
  result: EngineDesignResponse | null;
  isLoading: boolean;
}

const TABS: { id: WorkspaceTab; label: string; icon: string; needsResult?: boolean }[] = [
  { id: "schematic", label: "Schematic", icon: "diagram-tree" },
  { id: "dashboard", label: "Dashboard", icon: "dashboard", needsResult: true },
  { id: "contour",   label: "Contour + GD", icon: "timeline-line-chart", needsResult: true },
  { id: "3d",        label: "3D View", icon: "cube", needsResult: true },
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
            disabled={t.needsResult && !result && !isLoading}
            style={{ opacity: t.needsResult && !result && !isLoading ? 0.35 : 1 }}
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

        {!isLoading && workspaceTab === "schematic" && (
          <SchematicView config={config} result={result} />
        )}

        {!isLoading && workspaceTab === "dashboard" && result && (
          <div style={{ padding: 16 }}>
            <PlotlyRenderer figureJson={result.figure_dashboard} height={480} />
          </div>
        )}

        {!isLoading && workspaceTab === "dashboard" && !result && <EmptyPane tab="dashboard" />}

        {!isLoading && workspaceTab === "contour" && result && (
          <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 16 }}>
            <PlotlyRenderer figureJson={result.figure_contour} height={320} />
            <PlotlyRenderer figureJson={result.figure_gas_dynamics} height={320} />
          </div>
        )}

        {!isLoading && workspaceTab === "contour" && !result && <EmptyPane tab="contour" />}

        {!isLoading && workspaceTab === "3d" && result && result.figure_3d && (
          <div style={{ padding: 16 }}>
            <PlotlyRenderer figureJson={result.figure_3d} height={520} />
          </div>
        )}

        {!isLoading && workspaceTab === "3d" && result && !result.figure_3d && (
          <div className="empty-state">
            <div className="empty-state-icon">
              <Icon icon="cube" size={32} />
            </div>
            <div className="empty-state-text">
              3D view is not available for this design. Run with cooling enabled to generate the 3D model.
            </div>
          </div>
        )}

        {!isLoading && workspaceTab === "3d" && !result && <EmptyPane tab="3d" />}
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
      {/* Animated engine schematic skeleton */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 4,
        padding: 32,
      }}>
        {/* Pulsing bars simulating computation */}
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

function EmptyPane({ tab }: { tab: string }) {
  const msgs: Record<string, string> = {
    dashboard: "Run a design to see the performance dashboard.",
    contour: "Run a design to see the nozzle contour and gas dynamics.",
    "3d": "Run a design to generate the 3D nozzle model.",
  };
  return (
    <div className="empty-state">
      <div className="empty-state-icon">
        <Icon icon="play" size={36} />
      </div>
      <div className="empty-state-text">
        {msgs[tab] ?? "Run a design first."}
      </div>
    </div>
  );
}
