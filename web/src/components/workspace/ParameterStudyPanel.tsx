import { useState } from "react";
import { Button, Icon } from "@blueprintjs/core";
import { PlotlyRenderer } from "../plots/PlotlyRenderer";
import { useParameterStudyMutation } from "../../api/engine";
import type { EngineConfigRequest } from "../../types/engine";

interface ParameterStudyPanelProps {
  config: EngineConfigRequest;
}

export function ParameterStudyPanel({ config }: ParameterStudyPanelProps) {
  const [figureJson, setFigureJson] = useState<string | null>(null);
  const [ranFor, setRanFor] = useState<string | null>(null);
  const mutation = useParameterStudyMutation();

  async function handleRun() {
    try {
      const result = await mutation.mutateAsync(config);
      if (result.figure_study) {
        setFigureJson(result.figure_study);
        setRanFor(`${config.fuel} / ${config.oxidizer} · Pc ${config.pc_bar} bar · O/F ${config.mr}`);
      }
    } catch {
      // error displayed below
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Header bar */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 16,
        padding: "12px 16px",
        borderBottom: "1px solid var(--border)",
        flexShrink: 0,
      }}>
        <div style={{ flex: 1 }}>
          <div style={{
            fontSize: 11,
            color: "var(--text-secondary)",
            fontFamily: "var(--font-mono)",
          }}>
            Sweep Isp, C* and CF over O/F ratio, chamber pressure, and expansion ratio.
            Uses the current saved configuration as the design point.
          </div>
        </div>
        <Button
          intent="primary"
          icon="regression-chart"
          onClick={handleRun}
          loading={mutation.isPending}
          small
        >
          Run Parameter Study
        </Button>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: "auto" }}>
        {mutation.isPending && (
          <div style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            height: "100%",
            gap: 12,
          }}>
            <div style={{
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              color: "var(--text-secondary)",
              letterSpacing: "0.1em",
            }}>
              RUNNING CEA SWEEPS…
            </div>
            <div style={{ fontSize: 10, color: "var(--text-muted)" }}>
              ~40 points × 3 sweeps — takes 3–8 seconds
            </div>
          </div>
        )}

        {mutation.isError && !mutation.isPending && (
          <div style={{
            padding: 24,
            color: "var(--danger)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
          }}>
            <Icon icon="warning-sign" size={14} style={{ marginRight: 8 }} />
            Parameter study failed. Check that the engine configuration is valid and the API is running.
          </div>
        )}

        {!mutation.isPending && figureJson && (
          <div style={{ padding: 16 }}>
            {ranFor && (
              <div style={{
                marginBottom: 8,
                fontSize: 10,
                color: "var(--text-muted)",
                fontFamily: "var(--font-mono)",
              }}>
                <Icon icon="tick-circle" size={10} color="var(--green)" style={{ marginRight: 6 }} />
                {ranFor}
              </div>
            )}
            <PlotlyRenderer figureJson={figureJson} height={680} />
          </div>
        )}

        {!mutation.isPending && !figureJson && !mutation.isError && (
          <div className="empty-state">
            <div className="empty-state-icon">
              <Icon icon="regression-chart" size={36} />
            </div>
            <div className="empty-state-text">
              Click "Run Parameter Study" to sweep Isp, C* and CF over O/F ratio,
              chamber pressure, and expansion ratio.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
