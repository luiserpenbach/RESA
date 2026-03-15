import { Outlet } from "react-router-dom";
import { useUiStore } from "../../store/uiStore";
import { NavigationSidebar } from "./NavigationSidebar";

/**
 * App shell: NavigationSidebar | 3-panel grid (TopBar, LeftPanel, Workspace, RightPanel, StatusBar)
 */
export function AppShell() {
  const { sidebarCollapsed, rightPanelCollapsed, navCollapsed } = useUiStore();

  const gridCols = [
    sidebarCollapsed ? "0px" : "var(--panel-left)",
    "1fr",
    rightPanelCollapsed ? "0px" : "var(--panel-right)",
  ].join(" ");

  return (
    <div className="app-root">
      <NavigationSidebar collapsed={navCollapsed} />
      <div
        className="app-shell"
        style={{
          gridTemplateColumns: gridCols,
          transition: "grid-template-columns 180ms ease",
        }}
      >
        <Outlet />
      </div>
    </div>
  );
}
