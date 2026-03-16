import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { useUiStore } from "../../store/uiStore";
import { NavigationSidebar } from "./NavigationSidebar";
import { MethodsPanel } from "../common/MethodsPanel";
import { MODULE_DOCS } from "../../data/moduleDocs";

interface ErrorBoundaryState {
  hasError: boolean;
  message: string;
}

class PageErrorBoundary extends Component<{ children: ReactNode }, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false, message: "" };

  static getDerivedStateFromError(err: unknown): ErrorBoundaryState {
    return { hasError: true, message: err instanceof Error ? err.message : String(err) };
  }

  componentDidCatch(err: Error, info: ErrorInfo) {
    console.error("PageErrorBoundary caught:", err, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <>
          <div style={{ gridColumn: "1/-1", gridRow: 1, background: "var(--bg-panel)", borderBottom: "1px solid var(--border-subtle)", display: "flex", alignItems: "center", padding: "0 18px", gap: 12 }}>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 13, fontWeight: 500, color: "var(--accent-bright)", letterSpacing: "0.18em" }}>RESA</span>
          </div>
          <div style={{ gridColumn: 1, gridRow: 2, borderRight: "1px solid var(--border-subtle)" }} />
          <div style={{ gridColumn: 2, gridRow: 2, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)" }}>Page Error</div>
            <div style={{ fontSize: 12, color: "var(--text-muted)", maxWidth: 400, textAlign: "center", fontFamily: "var(--font-mono)", wordBreak: "break-word" }}>
              {this.state.message}
            </div>
            <button
              style={{ marginTop: 8, padding: "6px 16px", fontSize: 12, background: "var(--accent)", color: "#fff", border: "none", borderRadius: 4, cursor: "pointer" }}
              onClick={() => this.setState({ hasError: false, message: "" })}
            >
              Retry
            </button>
          </div>
          <div style={{ gridColumn: 3, gridRow: 2, borderLeft: "1px solid var(--border-subtle)" }} />
          <div style={{ gridColumn: "1/-1", gridRow: 3, background: "var(--accent)" }} />
        </>
      );
    }
    return this.props.children;
  }
}

/**
 * App shell: NavigationSidebar | 3-panel grid (TopBar, LeftPanel, Workspace, RightPanel, StatusBar)
 */
export function AppShell() {
  const { sidebarCollapsed, rightPanelCollapsed, navCollapsed, methodsPanelOpen } = useUiStore();
  const location = useLocation();
  const hasModuleDocs = !!MODULE_DOCS[location.pathname];

  const gridCols = [
    sidebarCollapsed ? "0px" : "var(--panel-left)",
    "1fr",
    rightPanelCollapsed ? "0px" : "var(--panel-right)",
  ].join(" ");

  const methodsHeight = methodsPanelOpen && hasModuleDocs ? 190 : 0;

  return (
    <div className="app-root">
      <NavigationSidebar collapsed={navCollapsed} />
      <div
        className="app-shell"
        style={{
          gridTemplateColumns: gridCols,
          gridTemplateRows: `var(--topbar-height) 1fr ${methodsHeight}px var(--statusbar-height)`,
          transition: "grid-template-columns 180ms ease, grid-template-rows 200ms ease",
        }}
      >
        <PageErrorBoundary>
          <Outlet />
        </PageErrorBoundary>
        <MethodsPanel />
      </div>
    </div>
  );
}
