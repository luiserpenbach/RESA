import { useEffect } from "react";
import { Icon } from "@blueprintjs/core";
import { useUiStore } from "../../store/uiStore";
import { useEngineStore } from "../../store/engineStore";

interface TopBarProps {
  onRunDesign: () => void;
  isRunning: boolean;
}

export function TopBar({ onRunDesign, isRunning }: TopBarProps) {
  const { toggleCmdPalette, setCmdPaletteOpen, toggleSidebar, toggleRightPanel } =
    useUiStore();
  const { activeConfig, lastDesignResult } = useEngineStore();

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        toggleCmdPalette();
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        if (!isRunning) onRunDesign();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [toggleCmdPalette, onRunDesign, isRunning]);

  const hasResult = !!lastDesignResult;
  const hasWarnings = (lastDesignResult?.warnings?.length ?? 0) > 0;

  let statusClass = "idle";
  let statusLabel = "READY";
  if (isRunning) { statusClass = "running"; statusLabel = "COMPUTING"; }
  else if (hasResult && hasWarnings) { statusClass = "warning"; statusLabel = "WARNING"; }
  else if (hasResult) { statusClass = "nominal"; statusLabel = "NOMINAL"; }

  return (
    <header className="app-topbar">
      {/* Logo */}
      <div className="topbar-logo">
        <div className="topbar-logo-dot" />
        RESA
      </div>

      {/* Breadcrumb */}
      <div className="topbar-breadcrumb">
        <span className="proj-name">{activeConfig.engine_name || "Unnamed Engine"}</span>
        <span className="sep">/</span>
        <span className="page-name">Engine Design</span>
      </div>

      {/* Status */}
      <div className={`topbar-status ${statusClass}`}>
        <div className={`status-dot ${isRunning ? "pulse" : ""}`} />
        {statusLabel}
      </div>

      <div className="topbar-spacer" />

      <div className="topbar-actions">
        {/* Panel toggles */}
        <button className="icon-btn" onClick={toggleSidebar} title="Toggle parameters panel">
          <Icon icon="panel-stats" size={14} />
        </button>
        <button className="icon-btn" onClick={toggleRightPanel} title="Toggle metrics panel">
          <Icon icon="panel-table" size={14} />
        </button>

        {/* Divider */}
        <div style={{ width: 1, height: 20, background: "var(--border-subtle)", margin: "0 4px" }} />

        {/* Command palette */}
        <button
          className="cmd-trigger"
          onClick={() => setCmdPaletteOpen(true)}
          title="Open command palette (⌘K)"
        >
          <Icon icon="search" size={12} />
          <span>Commands</span>
          <span className="kbd">⌘K</span>
        </button>

        {/* Run button */}
        <button
          className={`run-btn ${isRunning ? "is-running" : ""}`}
          onClick={onRunDesign}
          disabled={isRunning}
          title="Run engine design (⌘↵)"
        >
          <Icon icon={isRunning ? "dot" : "play"} size={12} />
          {isRunning ? "COMPUTING…" : "RUN DESIGN"}
        </button>
      </div>
    </header>
  );
}
