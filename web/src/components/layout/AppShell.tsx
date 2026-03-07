import { Outlet } from "react-router-dom";
import { useUiStore } from "../../store/uiStore";

/**
 * 3-panel grid shell: TopBar | LeftPanel | Workspace | RightPanel | StatusBar
 * The actual panel content is injected by the page (EnginePage) via context/props.
 * For the workspace routes, EnginePage renders all three panels directly inside the grid.
 */
export function AppShell() {
  const { sidebarCollapsed, rightPanelCollapsed } = useUiStore();

  const gridCols = [
    sidebarCollapsed ? "0px" : "var(--panel-left)",
    "1fr",
    rightPanelCollapsed ? "0px" : "var(--panel-right)",
  ].join(" ");

  return (
    <div
      className="app-shell"
      style={{
        gridTemplateColumns: gridCols,
        transition: "grid-template-columns 180ms ease",
      }}
    >
      <Outlet />
    </div>
  );
}
