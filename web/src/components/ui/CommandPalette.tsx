import { useEffect, useRef, useState } from "react";
import { Icon } from "@blueprintjs/core";
import { useNavigate } from "react-router-dom";
import { useUiStore } from "../../store/uiStore";

interface Command {
  id: string;
  label: string;
  desc: string;
  icon: string;
  shortcut?: string;
  action: () => void;
  group?: string;
}

interface CommandPaletteProps {
  onRunDesign: () => void;
  onValidate: () => void;
  onExportYaml: () => void;
}

export function CommandPalette({ onRunDesign, onValidate, onExportYaml }: CommandPaletteProps) {
  const { cmdPaletteOpen, setCmdPaletteOpen, setWorkspaceTab } = useUiStore();
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [focusIdx, setFocusIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const allCommands: Command[] = [
    {
      id: "run",
      label: "Run Engine Design",
      desc: "Solve combustion, geometry and cooling",
      icon: "play",
      shortcut: "⌘↵",
      action: () => { onRunDesign(); setCmdPaletteOpen(false); },
      group: "Actions",
    },
    {
      id: "validate",
      label: "Validate Configuration",
      desc: "Check configuration for errors and warnings",
      icon: "endorsed",
      action: () => { onValidate(); setCmdPaletteOpen(false); },
      group: "Actions",
    },
    {
      id: "export-yaml",
      label: "Export YAML",
      desc: "Download engine configuration as YAML",
      icon: "export",
      action: () => { onExportYaml(); setCmdPaletteOpen(false); },
      group: "Actions",
    },
    {
      id: "view-dashboard",
      label: "View Dashboard",
      desc: "Switch to engine design dashboard (contour + gas dynamics)",
      icon: "timeline-line-chart",
      action: () => { setWorkspaceTab("dashboard"); setCmdPaletteOpen(false); },
      group: "View",
    },
    {
      id: "view-parameter-study",
      label: "View Parameter Study",
      desc: "Switch to Isp / C* parametric sweeps",
      icon: "regression-chart",
      action: () => { setWorkspaceTab("parameter_study"); setCmdPaletteOpen(false); },
      group: "View",
    },
    {
      id: "view-suggested-values",
      label: "View Suggested Values",
      desc: "Switch to empirical L* and CR reference charts",
      icon: "book",
      action: () => { setWorkspaceTab("suggested_values"); setCmdPaletteOpen(false); },
      group: "View",
    },
    {
      id: "nav-engine",
      label: "Go to Engine Design",
      desc: "Open the engine design workspace",
      icon: "flame",
      action: () => { navigate("/engine"); setCmdPaletteOpen(false); },
      group: "Navigate",
    },
    {
      id: "nav-monte-carlo",
      label: "Go to Monte Carlo",
      desc: "Open uncertainty analysis module",
      icon: "scatter-plot",
      action: () => { navigate("/monte-carlo"); setCmdPaletteOpen(false); },
      group: "Navigate",
    },
    {
      id: "nav-optimization",
      label: "Go to Optimization",
      desc: "Open design optimization module",
      icon: "trending-up",
      action: () => { navigate("/optimization"); setCmdPaletteOpen(false); },
      group: "Navigate",
    },
    {
      id: "nav-injector",
      label: "Go to Injector Design",
      desc: "Open swirl injector design module",
      icon: "pivot-table",
      action: () => { navigate("/injector"); setCmdPaletteOpen(false); },
      group: "Navigate",
    },
  ];

  const filtered = query.trim()
    ? allCommands.filter(
        (c) =>
          c.label.toLowerCase().includes(query.toLowerCase()) ||
          c.desc.toLowerCase().includes(query.toLowerCase())
      )
    : allCommands;

  useEffect(() => {
    setFocusIdx(0);
  }, [query]);

  useEffect(() => {
    if (cmdPaletteOpen) {
      setQuery("");
      setFocusIdx(0);
      setTimeout(() => inputRef.current?.focus(), 30);
    }
  }, [cmdPaletteOpen]);

  useEffect(() => {
    if (!cmdPaletteOpen) return;

    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        setCmdPaletteOpen(false);
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setFocusIdx((i) => Math.min(i + 1, filtered.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setFocusIdx((i) => Math.max(i - 1, 0));
      } else if (e.key === "Enter") {
        e.preventDefault();
        filtered[focusIdx]?.action();
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [cmdPaletteOpen, filtered, focusIdx, setCmdPaletteOpen]);

  if (!cmdPaletteOpen) return null;

  // Group commands
  const groups: Record<string, Command[]> = {};
  filtered.forEach((c) => {
    const g = c.group ?? "Other";
    if (!groups[g]) groups[g] = [];
    groups[g].push(c);
  });

  let globalIdx = 0;

  return (
    <div className="cmd-overlay" onClick={() => setCmdPaletteOpen(false)}>
      <div className="cmd-palette" onClick={(e) => e.stopPropagation()}>
        {/* Input */}
        <div className="cmd-palette-input-row">
          <Icon icon="search" size={16} className="cmd-palette-icon" />
          <input
            ref={inputRef}
            className="cmd-palette-input"
            placeholder="Type a command or search…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoComplete="off"
            spellCheck={false}
          />
          {query && (
            <button
              className="icon-btn"
              style={{ width: 22, height: 22 }}
              onClick={() => setQuery("")}
            >
              <Icon icon="cross" size={10} />
            </button>
          )}
        </div>

        {/* Results */}
        <div className="cmd-palette-results">
          {filtered.length === 0 ? (
            <div style={{ padding: "20px 16px", textAlign: "center", color: "var(--text-muted)", fontSize: 12 }}>
              No commands match "{query}"
            </div>
          ) : (
            Object.entries(groups).map(([groupName, cmds]) => (
              <div key={groupName}>
                {!query && (
                  <div style={{
                    padding: "6px 16px 3px",
                    fontSize: 9,
                    fontWeight: 600,
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    color: "var(--text-muted)",
                  }}>
                    {groupName}
                  </div>
                )}
                {cmds.map((cmd) => {
                  const idx = globalIdx++;
                  return (
                    <div
                      key={cmd.id}
                      className={`cmd-item ${idx === focusIdx ? "focused" : ""}`}
                      onClick={cmd.action}
                      onMouseEnter={() => setFocusIdx(idx)}
                    >
                      <div className="cmd-item-icon-wrap">
                        <Icon icon={cmd.icon as never} size={13} />
                      </div>
                      <div className="cmd-item-text">
                        <div className="cmd-item-label">{cmd.label}</div>
                        <div className="cmd-item-desc">{cmd.desc}</div>
                      </div>
                      {cmd.shortcut && (
                        <div className="cmd-item-shortcut">{cmd.shortcut}</div>
                      )}
                    </div>
                  );
                })}
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="cmd-palette-footer">
          <span><span className="kbd">↑↓</span> navigate</span>
          <span><span className="kbd">↵</span> run</span>
          <span><span className="kbd">Esc</span> close</span>
        </div>
      </div>
    </div>
  );
}
