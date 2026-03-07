import { Icon } from "@blueprintjs/core";
import { useEngineStore } from "../../store/engineStore";
import { useUiStore } from "../../store/uiStore";

interface StatusBarProps {
  isRunning: boolean;
  error: string | null;
}

function formatAgo(ts: number): string {
  const s = Math.round((Date.now() - ts) / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  return `${Math.round(s / 60)}m ago`;
}

export function StatusBar({ isRunning, error }: StatusBarProps) {
  const { lastDesignResult } = useEngineStore();
  const { lastRunTime, lastRunDuration } = useUiStore();

  const warnings = lastDesignResult?.warnings?.length ?? 0;
  const r = lastDesignResult;

  return (
    <footer className="app-statusbar">
      {/* Connection / state */}
      <div className="statusbar-item">
        <Icon icon={isRunning ? "dot" : error ? "cross" : r ? "tick-circle" : "circle"} size={10} />
        <span>
          {isRunning ? "COMPUTING" : error ? "ERROR" : r ? "NOMINAL" : "READY"}
        </span>
      </div>

      {lastRunTime && (
        <div className="statusbar-item">
          <Icon icon="time" size={10} />
          <span>{formatAgo(lastRunTime)}</span>
        </div>
      )}

      {lastRunDuration != null && (
        <div className="statusbar-item mono">
          {(lastRunDuration / 1000).toFixed(1)}s
        </div>
      )}

      {r && warnings > 0 && (
        <div className="statusbar-item" style={{ color: "rgba(245,166,35,0.9)" }}>
          <Icon icon="warning-sign" size={10} />
          <span>{warnings} warning{warnings !== 1 ? "s" : ""}</span>
        </div>
      )}

      {r && (
        <>
          <div className="statusbar-item mono">
            Isp {r.isp_vac?.toFixed(1)} s
          </div>
          <div className="statusbar-item mono">
            Pc {r.combustion?.pc_bar?.toFixed(1) ?? "—"} bar
          </div>
          <div className="statusbar-item mono">
            MR {r.combustion?.mr?.toFixed(2) ?? "—"}
          </div>
          <div className="statusbar-item mono">
            ṁ {r.massflow_total?.toFixed(3)} kg/s
          </div>
        </>
      )}

      {error && (
        <div className="statusbar-item" style={{ color: "rgba(232,64,64,0.9)" }}>
          {error.length > 80 ? error.slice(0, 80) + "…" : error}
        </div>
      )}

      <div className="statusbar-spacer" />

      <div className="statusbar-item">
        <Icon icon="flame" size={10} />
        RESA v2.0.0
      </div>
    </footer>
  );
}
