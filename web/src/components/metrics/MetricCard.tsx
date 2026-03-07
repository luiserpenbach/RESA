import { Card, Intent } from "@blueprintjs/core";

interface MetricCardProps {
  label: string;
  value: string | number | null | undefined;
  unit?: string;
  intent?: Intent;
  subtext?: string;
}

/**
 * A compact card showing a single KPI value. Uses RESA dark theme colors.
 */
export function MetricCard({
  label,
  value,
  unit,
  intent,
  subtext,
}: MetricCardProps) {
  const displayValue = value != null ? String(value) : "—";

  const intentColors: Record<string, string> = {
    success: "#3ddc84",
    warning: "#f6ad55",
    danger: "#fc8181",
    primary: "#7ba7cc",
    none: "#e8f4fd",
  };
  const valueColor = intent ? (intentColors[intent] ?? "#e8f4fd") : "#e8f4fd";

  return (
    <Card
      style={{
        background: "#0d2137",
        border: "1px solid #1e3a5f",
        padding: "12px 16px",
        minWidth: 120,
      }}
    >
      <div
        style={{
          fontSize: "11px",
          fontWeight: 600,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: "#7ba7cc",
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: "22px",
          fontWeight: 700,
          color: valueColor,
          lineHeight: 1.2,
        }}
      >
        {displayValue}
        {unit && (
          <span
            style={{ fontSize: "13px", color: "#7ba7cc", marginLeft: 4 }}
          >
            {unit}
          </span>
        )}
      </div>
      {subtext && (
        <div style={{ fontSize: "11px", color: "#7ba7cc", marginTop: 2 }}>
          {subtext}
        </div>
      )}
    </Card>
  );
}
