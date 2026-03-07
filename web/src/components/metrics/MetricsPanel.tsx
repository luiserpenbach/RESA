import { useState } from "react";
import { Icon } from "@blueprintjs/core";
import { AnimatedValue } from "./AnimatedValue";
import { exportYaml } from "../../api/engine";
import type { EngineDesignResponse, EngineConfigRequest } from "../../types/engine";

interface MetricsPanelProps {
  result: EngineDesignResponse | null;
  config: EngineConfigRequest;
  isLoading: boolean;
}

/** Collapsible section in the right panel. */
function Section({
  label,
  defaultOpen = true,
  children,
  badge,
}: {
  label: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
  badge?: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="panel-section" style={{ borderBottom: "1px solid var(--border-subtle)" }}>
      <div
        className={`panel-section-header ${open ? "open" : ""}`}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="panel-section-label">{label}</span>
        {badge}
        <Icon
          icon="chevron-right"
          size={12}
          className="panel-section-chevron"
        />
      </div>
      {open && <div className="panel-section-body" style={{ padding: 0 }}>{children}</div>}
    </div>
  );
}

/** Single metric row with animated value. */
function MRow({
  name,
  value,
  unit,
  decimals = 1,
  status,
}: {
  name: string;
  value: number | null | undefined;
  unit: string;
  decimals?: number;
  status?: "good" | "warn" | "danger" | "neutral";
}) {
  return (
    <div className="metric-row">
      <span className="metric-name">{name}</span>
      <span className="metric-val">
        <AnimatedValue value={value ?? null} decimals={decimals} />
        <span className="metric-unit">{unit}</span>
      </span>
      {status && <div className={`metric-badge ${status}`} />}
    </div>
  );
}

/** Static (non-animated) string metric row. */
function SRow({ name, value }: { name: string; value: string | number | null | undefined }) {
  return (
    <div className="metric-row">
      <span className="metric-name">{name}</span>
      <span className="metric-val" style={{ fontSize: 12 }}>
        {value ?? "—"}
      </span>
    </div>
  );
}

export function MetricsPanel({ result: r, config, isLoading }: MetricsPanelProps) {
  const hasThermal = r?.max_wall_temp != null;

  // Determine Isp status relative to typical ranges
  function ispStatus(isp: number | undefined) {
    if (!isp) return "neutral";
    if (isp > 300) return "good";
    if (isp > 250) return "warn";
    return "danger";
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {/* Panel header */}
      <div style={{
        flexShrink: 0,
        padding: "7px 12px",
        borderBottom: "1px solid var(--border-subtle)",
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}>
        <span style={{
          fontSize: 10, fontWeight: 600, letterSpacing: "0.12em",
          textTransform: "uppercase", color: "var(--text-muted)",
        }}>
          Analysis Results
        </span>
        {r && (
          <span style={{
            marginLeft: "auto", fontSize: 9,
            fontFamily: "var(--font-mono)",
            color: "var(--green)",
            letterSpacing: "0.08em",
          }}>
            ● COMPUTED
          </span>
        )}
        {isLoading && (
          <span style={{
            marginLeft: "auto", fontSize: 9,
            fontFamily: "var(--font-mono)",
            color: "var(--accent-bright)",
            letterSpacing: "0.08em",
            animation: "dot-pulse 1.4s ease-in-out infinite",
          }}>
            ● SOLVING
          </span>
        )}
      </div>

      <div className="panel-scroll">
        {/* ── PERFORMANCE ─────────────────────────── */}
        <Section label="Performance">
          {isLoading ? (
            <SkeletonRows n={5} />
          ) : r ? (
            <div className="metric-block">
              <MRow name="Isp (vacuum)" value={r.isp_vac} unit="s" decimals={1} status={ispStatus(r.isp_vac)} />
              <MRow name="Isp (sea level)" value={r.isp_sea} unit="s" decimals={1} />
              <MRow name="Thrust (vac)" value={r.thrust_vac ? r.thrust_vac / 1000 : null} unit="kN" decimals={3} status="good" />
              <MRow name="Thrust (sea)" value={r.thrust_sea ? r.thrust_sea / 1000 : null} unit="kN" decimals={3} />
              <MRow name="Mass flow" value={r.massflow_total} unit="kg/s" decimals={4} />
              <MRow name="C* efficiency" value={r.combustion?.cstar != null ? 100 : null} unit="%" decimals={1} />
            </div>
          ) : (
            <EmptyMetric />
          )}
        </Section>

        {/* ── GEOMETRY ────────────────────────────── */}
        <Section label="Geometry">
          {isLoading ? (
            <SkeletonRows n={4} />
          ) : r ? (
            <div className="metric-block">
              <MRow name="Throat Ø" value={r.dt_mm} unit="mm" decimals={1} />
              <MRow name="Exit Ø" value={r.de_mm} unit="mm" decimals={1} />
              <MRow name="Length" value={r.length_mm} unit="mm" decimals={0} />
              <MRow name="Expansion ratio" value={r.expansion_ratio} unit="" decimals={2} />
            </div>
          ) : (
            <EmptyMetric />
          )}
        </Section>

        {/* ── COMBUSTION ──────────────────────────── */}
        <Section label="Combustion" defaultOpen={true}>
          {isLoading ? (
            <SkeletonRows n={6} />
          ) : r?.combustion ? (
            <div className="metric-block">
              <MRow name="Chamber pressure" value={r.combustion.pc_bar} unit="bar" decimals={1} />
              <MRow name="Mixture ratio" value={r.combustion.mr} unit="O/F" decimals={3} />
              <MRow name="C*" value={r.combustion.cstar} unit="m/s" decimals={0} />
              <MRow name="Flame temp" value={r.combustion.T_combustion} unit="K" decimals={0} status={
                r.combustion.T_combustion ? r.combustion.T_combustion > 3600 ? "warn" : "good" : "neutral"
              } />
              <MRow name="γ (gamma)" value={r.combustion.gamma} unit="" decimals={4} />
              <MRow name="Mol weight" value={r.combustion.mw} unit="g/mol" decimals={2} />
              <SRow name="Mach at exit" value={r.combustion.mach_exit?.toFixed(3)} />
            </div>
          ) : (
            <EmptyMetric />
          )}
        </Section>

        {/* ── THERMAL (cooling) ───────────────────── */}
        {(hasThermal || isLoading) && (
          <Section label="Thermal Analysis">
            {isLoading ? (
              <SkeletonRows n={3} />
            ) : (
              <div className="metric-block">
                <MRow
                  name="Max wall temp"
                  value={r?.max_wall_temp}
                  unit="K"
                  decimals={0}
                  status={r?.max_wall_temp ? r.max_wall_temp > 900 ? "warn" : "good" : "neutral"}
                />
                <MRow name="Coolant ΔP" value={r?.pressure_drop_bar} unit="bar" decimals={2} />
                <MRow name="Coolant outlet T" value={r?.outlet_temp_k} unit="K" decimals={1} />
              </div>
            )}
          </Section>
        )}

        {/* ── WARNINGS ────────────────────────────── */}
        {r?.warnings && r.warnings.length > 0 && (
          <Section
            label="Warnings"
            defaultOpen={true}
            badge={
              <span style={{
                fontSize: 10, fontFamily: "var(--font-mono)",
                color: "var(--amber)", marginLeft: 4,
              }}>
                {r.warnings.length}
              </span>
            }
          >
            <div>
              {r.warnings.map((w, i) => (
                <div key={i} className="warning-item">
                  <Icon icon="warning-sign" size={10} style={{ marginTop: 1, flexShrink: 0, color: "var(--amber)" }} />
                  <span>{w}</span>
                </div>
              ))}
            </div>
          </Section>
        )}

        {/* ── CONFIG SUMMARY ──────────────────────── */}
        <Section label="Configuration" defaultOpen={false}>
          <div className="metric-block">
            <SRow name="Fuel" value={config.fuel} />
            <SRow name="Oxidizer" value={config.oxidizer} />
            <SRow name="Target thrust" value={`${config.thrust_n} N`} />
            <SRow name="Target Pc" value={`${config.pc_bar} bar`} />
            <SRow name="Target MR" value={config.mr} />
            <SRow name="Nozzle type" value={config.nozzle_type} />
            <SRow name="Coolant" value={config.coolant_name?.split("::")[1] ?? config.coolant_name} />
            <SRow name="Wall material" value={config.wall_material} />
          </div>
        </Section>
      </div>

      {/* Footer actions */}
      <div className="panel-footer">
        <button
          className="bp5-button bp5-dark"
          style={{ flex: 1, justifyContent: "center" }}
          onClick={async () => {
            try {
              const blob = await exportYaml(config);
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = `${config.engine_name ?? "engine"}.yaml`;
              a.click();
              URL.revokeObjectURL(url);
            } catch { /* silent */ }
          }}
          title="Export configuration as YAML"
        >
          <Icon icon="export" size={12} />
          Export YAML
        </button>

        {r && (
          <button
            className="bp5-button bp5-dark"
            style={{ flex: 1, justifyContent: "center" }}
            onClick={() => {
              if (!r.contour_x_mm || !r.contour_y_mm) return;
              const csv =
                "x_mm,r_mm\n" +
                r.contour_x_mm.map((x, i) => `${x},${r.contour_y_mm![i]}`).join("\n");
              const blob = new Blob([csv], { type: "text/csv" });
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = "nozzle_contour.csv";
              a.click();
              URL.revokeObjectURL(url);
            }}
            title="Export nozzle contour as CSV"
          >
            <Icon icon="download" size={12} />
            Contour CSV
          </button>
        )}
      </div>
    </div>
  );
}

function EmptyMetric() {
  return (
    <div style={{
      padding: "14px 12px",
      fontSize: 11,
      color: "var(--text-muted)",
      fontStyle: "italic",
    }}>
      Run a design to see results.
    </div>
  );
}

function SkeletonRows({ n }: { n: number }) {
  return (
    <div className="metric-block">
      {Array.from({ length: n }).map((_, i) => (
        <div key={i} className="metric-row">
          <div className="skeleton" style={{ width: "55%", height: 10 }} />
          <div className="skeleton" style={{ width: "28%", height: 10 }} />
        </div>
      ))}
    </div>
  );
}
