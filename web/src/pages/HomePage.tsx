import { Card, H3, H5, NonIdealState, Button } from "@blueprintjs/core";
import { useNavigate } from "react-router-dom";
import { MetricGrid } from "../components/metrics/MetricGrid";
import { useEngineStore } from "../store/engineStore";

export default function HomePage() {
  const navigate = useNavigate();
  const { lastDesignResult, activeConfig } = useEngineStore();

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <H3 style={{ color: "#e8f4fd", margin: 0 }}>
          Rocket Engine Sizing &amp; Analysis
        </H3>
        <p style={{ color: "#7ba7cc", marginTop: 4 }}>
          Professional liquid rocket engine preliminary design toolkit
        </p>
      </div>

      {/* Active engine strip */}
      {lastDesignResult ? (
        <div style={{ marginBottom: 24 }}>
          <H5 style={{ color: "#7ba7cc", marginBottom: 12 }}>
            Last Design:{" "}
            <span style={{ color: "#e8f4fd" }}>{activeConfig.engine_name}</span>
          </H5>
          <MetricGrid result={lastDesignResult} />
        </div>
      ) : (
        <Card
          style={{
            background: "#0d2137",
            border: "1px solid #1e3a5f",
            marginBottom: 24,
          }}
        >
          <NonIdealState
            icon="flame"
            title="No design results yet"
            description="Navigate to Engine Design to run your first analysis."
            action={
              <Button
                intent="primary"
                icon="arrow-right"
                onClick={() => navigate("/engine")}
              >
                Go to Engine Design
              </Button>
            }
          />
        </Card>
      )}

      {/* Quick navigation */}
      <H5 style={{ color: "#7ba7cc", marginBottom: 12 }}>Quick Access</H5>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: 12,
        }}
      >
        {[
          { label: "Engine Design", path: "/engine", icon: "🚀", desc: "Thrust chamber sizing" },
          { label: "Cooling Analysis", path: "/cooling", icon: "❄️", desc: "Regenerative cooling" },
          { label: "Monte Carlo", path: "/monte-carlo", icon: "📊", desc: "Uncertainty analysis" },
          { label: "Throttle Analysis", path: "/throttle", icon: "⚡", desc: "Throttle curve mapping" },
          { label: "Injector Design", path: "/injector", icon: "💧", desc: "Swirl injector sizing" },
          { label: "Projects", path: "/projects", icon: "📁", desc: "Design management" },
        ].map((item) => (
          <Card
            key={item.path}
            interactive
            onClick={() => navigate(item.path)}
            style={{
              background: "#0d2137",
              border: "1px solid #1e3a5f",
              cursor: "pointer",
            }}
          >
            <div style={{ fontSize: 24, marginBottom: 8 }}>{item.icon}</div>
            <div
              style={{ color: "#e8f4fd", fontWeight: 600, marginBottom: 4 }}
            >
              {item.label}
            </div>
            <div style={{ color: "#7ba7cc", fontSize: 13 }}>{item.desc}</div>
          </Card>
        ))}
      </div>
    </div>
  );
}
