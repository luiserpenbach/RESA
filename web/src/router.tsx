import { lazy, Suspense } from "react";
import {
  createBrowserRouter,
  type RouteObject,
} from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { NonIdealState, Spinner } from "@blueprintjs/core";

const HomePage = lazy(() => import("./pages/HomePage"));
const EnginePage = lazy(() => import("./pages/EnginePage"));

function ComingSoon({ name }: { name: string }) {
  return (
    <NonIdealState
      icon="time"
      title={name}
      description="This module is coming in Phase 2. Engine Design is available now."
    />
  );
}

function Loading() {
  return (
    <div style={{ display: "flex", justifyContent: "center", padding: 64 }}>
      <Spinner intent="primary" />
    </div>
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
      { index: true, element: withSuspense(<HomePage />) },
      { path: "engine", element: withSuspense(<EnginePage />) },
      { path: "cooling", element: <ComingSoon name="Cooling Analysis" /> },
      { path: "contour", element: <ComingSoon name="Nozzle Contour" /> },
      { path: "throttle", element: <ComingSoon name="Throttle Analysis" /> },
      { path: "analysis", element: <ComingSoon name="Off-Design Analysis" /> },
      { path: "monte-carlo", element: <ComingSoon name="Monte Carlo" /> },
      { path: "optimization", element: <ComingSoon name="Optimization" /> },
      { path: "injector", element: <ComingSoon name="Injector Design" /> },
      { path: "igniter", element: <ComingSoon name="Igniter Design" /> },
      { path: "tank", element: <ComingSoon name="Tank Simulation" /> },
      { path: "projects", element: <ComingSoon name="Projects" /> },
      { path: "settings", element: <ComingSoon name="Settings" /> },
    ],
  },
];

export const router = createBrowserRouter(routes);
