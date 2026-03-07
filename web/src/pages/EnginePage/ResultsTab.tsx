import { H5, Callout } from "@blueprintjs/core";
import { MetricGrid } from "../../components/metrics/MetricGrid";
import { PlotlyRenderer } from "../../components/plots/PlotlyRenderer";
import { LoadingOverlay } from "../../components/common/LoadingOverlay";
import { ErrorCallout } from "../../components/common/ErrorCallout";
import type { EngineDesignResponse } from "../../types/engine";

interface ResultsTabProps {
  result: EngineDesignResponse | null;
  isLoading: boolean;
  error: string | null;
}

export function ResultsTab({ result, isLoading, error }: ResultsTabProps) {
  if (isLoading) {
    return <LoadingOverlay label="Running engine design..." />;
  }

  if (error) {
    return <ErrorCallout title="Design Failed" message={error} />;
  }

  return (
    <div style={{ padding: "16px 0" }}>
      {result?.warnings && result.warnings.length > 0 && (
        <Callout intent="warning" title="Warnings" style={{ marginBottom: 16 }}>
          <ul style={{ margin: 0, paddingLeft: 20 }}>
            {result.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </Callout>
      )}

      <H5 style={{ color: "#7ba7cc", marginBottom: 12 }}>Performance Metrics</H5>
      <MetricGrid result={result} />

      {result?.cooling && (
        <>
          <H5 style={{ color: "#7ba7cc", margin: "20px 0 12px" }}>
            Thermal Analysis
          </H5>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))",
              gap: 10,
            }}
          >
            <div style={{ background: "#0d2137", border: "1px solid #1e3a5f", padding: "12px 16px", borderRadius: 2 }}>
              <div style={{ fontSize: 11, color: "#7ba7cc", textTransform: "uppercase", letterSpacing: "0.08em" }}>Max Wall Temp</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: "#e8f4fd" }}>
                {result.max_wall_temp?.toFixed(0)} <span style={{ fontSize: 13, color: "#7ba7cc" }}>K</span>
              </div>
            </div>
            <div style={{ background: "#0d2137", border: "1px solid #1e3a5f", padding: "12px 16px", borderRadius: 2 }}>
              <div style={{ fontSize: 11, color: "#7ba7cc", textTransform: "uppercase", letterSpacing: "0.08em" }}>Pressure Drop</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: "#e8f4fd" }}>
                {result.pressure_drop_bar?.toFixed(2)} <span style={{ fontSize: 13, color: "#7ba7cc" }}>bar</span>
              </div>
            </div>
            <div style={{ background: "#0d2137", border: "1px solid #1e3a5f", padding: "12px 16px", borderRadius: 2 }}>
              <div style={{ fontSize: 11, color: "#7ba7cc", textTransform: "uppercase", letterSpacing: "0.08em" }}>Outlet Temp</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: "#e8f4fd" }}>
                {result.outlet_temp_k?.toFixed(1)} <span style={{ fontSize: 13, color: "#7ba7cc" }}>K</span>
              </div>
            </div>
          </div>
        </>
      )}

      {result && (
        <>
          <H5 style={{ color: "#7ba7cc", margin: "20px 0 12px" }}>
            Engine Dashboard
          </H5>
          <PlotlyRenderer figureJson={result.figure_dashboard} height={500} />

          <H5 style={{ color: "#7ba7cc", margin: "20px 0 12px" }}>
            Nozzle Contour
          </H5>
          <PlotlyRenderer figureJson={result.figure_contour} height={350} />

          <H5 style={{ color: "#7ba7cc", margin: "20px 0 12px" }}>
            Gas Dynamics
          </H5>
          <PlotlyRenderer figureJson={result.figure_gas_dynamics} height={350} />

          {result.figure_3d && (
            <>
              <H5 style={{ color: "#7ba7cc", margin: "20px 0 12px" }}>
                3D Viewer
              </H5>
              <PlotlyRenderer figureJson={result.figure_3d} height={500} />
            </>
          )}
        </>
      )}

      {!result && (
        <div
          style={{
            textAlign: "center",
            padding: "60px 0",
            color: "#5c7d9e",
          }}
        >
          Run a design on the Configuration tab to see results here.
        </div>
      )}
    </div>
  );
}
