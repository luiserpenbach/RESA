import { MetricCard } from "./MetricCard";
import type { EngineDesignResponse } from "../../types/engine";

interface MetricGridProps {
  result: EngineDesignResponse | null;
}

/**
 * Responsive grid of 12 key performance metrics from EngineDesignResponse.
 */
export function MetricGrid({ result }: MetricGridProps) {
  if (!result) {
    return (
      <div style={{ color: "#7ba7cc", padding: "16px 0" }}>
        No design results yet. Run a design to see metrics.
      </div>
    );
  }

  const metrics = [
    {
      label: "Isp (Vacuum)",
      value: result.isp_vac.toFixed(1),
      unit: "s",
    },
    {
      label: "Isp (Sea Level)",
      value: result.isp_sea.toFixed(1),
      unit: "s",
    },
    {
      label: "Thrust (Vac)",
      value: result.thrust_vac.toFixed(0),
      unit: "N",
    },
    {
      label: "Thrust (SL)",
      value: result.thrust_sea.toFixed(0),
      unit: "N",
    },
    {
      label: "Chamber Pressure",
      value: result.pc_bar.toFixed(1),
      unit: "bar",
    },
    {
      label: "Mixture Ratio",
      value: result.mr.toFixed(2),
      unit: "O/F",
    },
    {
      label: "Mass Flow",
      value: result.massflow_total.toFixed(3),
      unit: "kg/s",
    },
    {
      label: "Throat Diameter",
      value: result.dt_mm.toFixed(2),
      unit: "mm",
    },
    {
      label: "Exit Diameter",
      value: result.de_mm.toFixed(2),
      unit: "mm",
    },
    {
      label: "Expansion Ratio",
      value: result.expansion_ratio.toFixed(2),
    },
    {
      label: "Engine Length",
      value: result.length_mm.toFixed(1),
      unit: "mm",
    },
    {
      label: "c*",
      value: result.combustion ? result.combustion.cstar.toFixed(1) : "—",
      unit: "m/s",
    },
  ];

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))",
        gap: 10,
      }}
    >
      {metrics.map((m) => (
        <MetricCard key={m.label} {...m} />
      ))}
    </div>
  );
}
