import { lazy, Suspense } from "react";
import { createBrowserRouter, type RouteObject } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { Icon } from "@blueprintjs/core";

const HomePage    = lazy(() => import("./pages/HomePage"));
const EnginePage  = lazy(() => import("./pages/EnginePage"));

function Loading() {
  return (
    <>
      {/* Placeholders that fill the 3 grid cells so layout doesn't collapse */}
      <div style={{ gridColumn: "1/-1", gridRow: 1, background: "var(--bg-panel)", borderBottom: "1px solid var(--border-subtle)" }} />
      <div style={{ gridColumn: 1, gridRow: 2, borderRight: "1px solid var(--border-subtle)" }} />
      <div style={{ gridColumn: 2, gridRow: 2, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{
          fontFamily: "var(--font-mono)", fontSize: 11,
          color: "var(--text-muted)", letterSpacing: "0.1em",
        }}>
          LOADING…
        </div>
      </div>
      <div style={{ gridColumn: 3, gridRow: 2, borderLeft: "1px solid var(--border-subtle)" }} />
      <div style={{ gridColumn: "1/-1", gridRow: 3, background: "var(--accent)" }} />
    </>
  );
}

function ComingSoon({ name }: { name: string }) {
  return (
    <>
      <div style={{ gridColumn: "1/-1", gridRow: 1, background: "var(--bg-panel)", borderBottom: "1px solid var(--border-subtle)", display: "flex", alignItems: "center", padding: "0 18px", gap: 12 }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 13, fontWeight: 500, color: "var(--accent-bright)", letterSpacing: "0.18em" }}>RESA</span>
        <span style={{ color: "var(--text-muted)", fontSize: 14 }}>/</span>
        <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>{name}</span>
      </div>
      <div style={{ gridColumn: 1, gridRow: 2, borderRight: "1px solid var(--border-subtle)" }} />
      <div style={{ gridColumn: 2, gridRow: 2, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16 }}>
        <div className="coming-soon-tag">Phase 2</div>
        <div className="coming-soon-title">{name}</div>
        <div className="coming-soon-desc">
          This module is coming in Phase 2. Engine Design workspace is available now.
        </div>
        <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
          <Icon icon="time" size={14} style={{ color: "var(--text-muted)" }} />
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Scheduled for upcoming release</span>
        </div>
      </div>
      <div style={{ gridColumn: 3, gridRow: 2, borderLeft: "1px solid var(--border-subtle)" }} />
      <div style={{ gridColumn: "1/-1", gridRow: 3, background: "var(--accent)" }} />
    </>
  );
}

function withSuspense(element: React.ReactNode) {
  return <Suspense fallback={<Loading />}>{element}</Suspense>;
}

const routes: RouteObject[] = [
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true,             element: withSuspense(<HomePage />) },
      { path: "engine",          element: withSuspense(<EnginePage />) },
      { path: "cooling",         element: <ComingSoon name="Cooling Analysis" /> },
      { path: "contour",         element: <ComingSoon name="Nozzle Contour" /> },
      { path: "throttle",        element: <ComingSoon name="Throttle Analysis" /> },
      { path: "analysis",        element: <ComingSoon name="Off-Design Analysis" /> },
      { path: "monte-carlo",     element: <ComingSoon name="Monte Carlo" /> },
      { path: "optimization",    element: <ComingSoon name="Optimization" /> },
      { path: "injector",        element: <ComingSoon name="Injector Design" /> },
      { path: "igniter",         element: <ComingSoon name="Igniter Design" /> },
      { path: "tank",            element: <ComingSoon name="Tank Simulation" /> },
      { path: "projects",        element: <ComingSoon name="Projects" /> },
      { path: "settings",        element: <ComingSoon name="Settings" /> },
    ],
  },
];

export const router = createBrowserRouter(routes);
