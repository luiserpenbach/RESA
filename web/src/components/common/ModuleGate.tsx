/**
 * Wraps a page and blocks access if prerequisite modules haven't been run.
 */
import { Icon } from "@blueprintjs/core";
import { Link } from "react-router-dom";
import { useDesignSessionStore } from "../../store/designSessionStore";
import type { ModuleName } from "../../types/session";

const MODULE_LABELS: Record<ModuleName, string> = {
  engine: "Engine Design",
  contour: "Nozzle Contour",
  cooling_channels: "Cooling Channels",
  cooling: "Cooling Analysis",
  wall_thickness: "Wall Thickness",
  performance: "Performance Maps",
  feed_system: "Feed System",
};

const MODULE_PATHS: Record<ModuleName, string> = {
  engine: "/engine",
  contour: "/contour",
  cooling_channels: "/cooling",
  cooling: "/cooling",
  wall_thickness: "/structural",
  performance: "/performance",
  feed_system: "/feed-system",
};

interface ModuleGateProps {
  requires: ModuleName[];
  children: React.ReactNode;
}

export function ModuleGate({ requires, children }: ModuleGateProps) {
  const moduleStatus = useDesignSessionStore((s) => s.moduleStatus);

  const unmet = requires.filter(
    (m) => moduleStatus[m] !== "completed" && moduleStatus[m] !== "stale"
  );

  if (unmet.length > 0) {
    return (
      <>
        {/* Top bar placeholder */}
        <div
          style={{
            gridColumn: "1/-1",
            gridRow: 1,
            background: "var(--bg-panel)",
            borderBottom: "1px solid var(--border-subtle)",
            display: "flex",
            alignItems: "center",
            padding: "0 18px",
            gap: 12,
          }}
        >
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 13,
              fontWeight: 500,
              color: "var(--accent-bright)",
              letterSpacing: "0.18em",
            }}
          >
            RESA
          </span>
        </div>

        {/* Left panel placeholder */}
        <div
          style={{
            gridColumn: 1,
            gridRow: 2,
            borderRight: "1px solid var(--border-subtle)",
          }}
        />

        {/* Center: dependency message */}
        <div
          style={{
            gridColumn: 2,
            gridRow: 2,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 16,
          }}
        >
          <Icon icon="lock" size={28} style={{ color: "var(--text-muted)" }} />
          <div
            style={{
              fontSize: 14,
              fontWeight: 600,
              color: "var(--text-primary)",
            }}
          >
            Prerequisites Required
          </div>
          <div
            style={{
              fontSize: 12,
              color: "var(--text-muted)",
              textAlign: "center",
              maxWidth: 320,
              lineHeight: 1.6,
            }}
          >
            Run the following modules first:
          </div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 8,
              marginTop: 4,
            }}
          >
            {unmet.map((m) => (
              <Link
                key={m}
                to={MODULE_PATHS[m]}
                style={{
                  fontSize: 12,
                  color: "var(--accent-bright)",
                  textDecoration: "none",
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                }}
              >
                <Icon icon="arrow-right" size={12} />
                {MODULE_LABELS[m]}
              </Link>
            ))}
          </div>
        </div>

        {/* Right panel placeholder */}
        <div
          style={{
            gridColumn: 3,
            gridRow: 2,
            borderLeft: "1px solid var(--border-subtle)",
          }}
        />

        {/* Status bar */}
        <div
          style={{
            gridColumn: "1/-1",
            gridRow: 3,
            background: "var(--accent)",
          }}
        />
      </>
    );
  }

  return <>{children}</>;
}
