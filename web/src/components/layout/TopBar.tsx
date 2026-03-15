import { useEffect } from "react";
import { Icon } from "@blueprintjs/core";
import { useLocation } from "react-router-dom";
import { useUiStore } from "../../store/uiStore";
import { useEngineStore } from "../../store/engineStore";

const PAGE_NAMES: Record<string, string> = {
  "/engine": "Engine Design",
  "/contour": "Nozzle Contour",
  "/cooling": "Cooling Design",
  "/structural": "Wall Thickness",
  "/performance": "Performance Maps",
  "/feed-system": "Feed System",
  "/monte-carlo": "Monte Carlo",
  "/optimization": "Optimization",
  "/injector": "Injector Design",
  "/igniter": "Igniter Design",
  "/tank": "Tank Simulation",
  "/projects": "Projects",
  "/settings": "Settings",
};

interface TopBarProps {
  onRunDesign: () => void;
  isRunning: boolean;
}

export function TopBar({ onRunDesign, isRunning }: TopBarProps) {
  const { toggleCmdPalette, setCmdPaletteOpen, toggleSidebar, toggleRightPanel, toggleNav } =
    useUiStore();
  const { activeConfig, lastDesignResult } = useEngineStore();
  const location = useLocation();
  const pageName = PAGE_NAMES[location.pathname] ?? "Engine Design";

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
      {/* Nav toggle */}
      <button className="icon-btn" onClick={toggleNav} title="Toggle navigation" style={{ marginLeft: 6 }}>
        <Icon icon="menu" size={16} />
      </button>

      {/* Breadcrumb */}
      <div className="topbar-breadcrumb">
        <span className="proj-name">{activeConfig.engine_name || "Unnamed Engine"}</span>
        <span className="sep">/</span>
        <span className="page-name">{pageName}</span>
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
