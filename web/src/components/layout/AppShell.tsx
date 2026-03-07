import { Outlet } from "react-router-dom";
import { NavigationSidebar } from "./NavigationSidebar";
import { useUiStore } from "../../store/uiStore";

export function AppShell() {
  const { sidebarCollapsed } = useUiStore();

  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        overflow: "hidden",
        background: "#0d1117",
      }}
    >
      {!sidebarCollapsed && <NavigationSidebar />}

      <main
        style={{
          flex: 1,
          overflowY: "auto",
          padding: 24,
          background: "#0d1117",
        }}
      >
        <Outlet />
      </main>
    </div>
  );
}
