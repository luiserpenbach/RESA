import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import { NonIdealState, Spinner } from "@blueprintjs/core";

interface PlotlyRendererProps {
  /** Serialized figure JSON from Python's fig.to_json() */
  figureJson: string | null | undefined;
  height?: number;
  loading?: boolean;
}

/**
 * Renders a Plotly figure from a JSON string produced by Python's fig.to_json().
 * Uses Plotly.react() for efficient updates.
 */
export function PlotlyRenderer({
  figureJson,
  height = 400,
  loading = false,
}: PlotlyRendererProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    if (!figureJson) return;

    let figure: { data: Plotly.Data[]; layout: Partial<Plotly.Layout> };
    try {
      figure = JSON.parse(figureJson) as typeof figure;
    } catch {
      return;
    }

    const layout: Partial<Plotly.Layout> = {
      ...figure.layout,
      autosize: true,
      // Ensure dark background is preserved
      paper_bgcolor: figure.layout?.paper_bgcolor ?? "#0d1117",
      plot_bgcolor: figure.layout?.plot_bgcolor ?? "#0d1117",
    };

    Plotly.react(containerRef.current, figure.data, layout, {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ["sendDataToCloud" as Plotly.ModeBarDefaultButtons],
    });
  }, [figureJson]);

  if (loading) {
    return (
      <div
        style={{
          height,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Spinner />
      </div>
    );
  }

  if (!figureJson) {
    return (
      <div style={{ height }}>
        <NonIdealState
          icon="chart"
          title="No data"
          description="Run a design to see the plot."
        />
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height, minHeight: height }}
    />
  );
}
