import type { EngineDesignResponse } from "../../types/engine";
import type { EngineConfigRequest } from "../../types/engine";

interface SchematicViewProps {
  config: EngineConfigRequest;
  result: EngineDesignResponse | null;
}

/**
 * SVG cross-section of a bell-nozzle rocket engine.
 * Parameterised by contraction ratio, expansion ratio, and geometry data.
 */
export function SchematicView({ config, result }: SchematicViewProps) {
  // ── geometry ────────────────────────────────────────────────────────
  const cr = Math.max(1.5, config.contraction_ratio ?? 4.0);
  const er = Math.max(1.2, result?.expansion_ratio ?? config.expansion_ratio ?? 4.0);

  const W = 880;
  const H = 300;
  const cy = H / 2; // centre axis y

  // Normalised radii (px)
  const throatR = 36;
  const chamberR = Math.min(throatR * Math.sqrt(cr), 110);
  const exitR = Math.min(throatR * Math.sqrt(er), 115);

  // x positions
  const injectX = 60;
  const chamberEndX = injectX + 175;
  const throatX = chamberEndX + 72;
  const exitX = Math.min(throatX + 60 + exitR * 5.5, W - 50);

  // Bezier control points for upper bell wall
  // Initial divergence ~27°, exit ~10°
  const bellCp1x = throatX + (exitX - throatX) * 0.28;
  const bellCp1y = cy - throatR - (exitR - throatR) * 0.55;
  const bellCp2x = throatX + (exitX - throatX) * 0.70;
  const bellCp2y = cy - exitR + (exitR - throatR) * 0.08;

  // Upper wall path
  const upperWall =
    `M ${injectX},${cy - chamberR} ` +
    `L ${chamberEndX},${cy - chamberR} ` +
    `L ${throatX},${cy - throatR} ` +
    `C ${bellCp1x},${bellCp1y} ${bellCp2x},${bellCp2y} ${exitX},${cy - exitR}`;

  // Lower wall path (mirror)
  const lowerWall =
    `M ${injectX},${cy + chamberR} ` +
    `L ${chamberEndX},${cy + chamberR} ` +
    `L ${throatX},${cy + throatR} ` +
    `C ${bellCp1x},${H - bellCp1y} ${bellCp2x},${H - bellCp2y} ${exitX},${cy + exitR}`;

  // Full interior fill path (closed)
  const interiorPath =
    `M ${injectX},${cy - chamberR} ` +
    `L ${chamberEndX},${cy - chamberR} ` +
    `L ${throatX},${cy - throatR} ` +
    `C ${bellCp1x},${bellCp1y} ${bellCp2x},${bellCp2y} ${exitX},${cy - exitR} ` +
    `L ${exitX},${cy + exitR} ` +
    `C ${bellCp2x},${H - bellCp2y} ${bellCp1x},${H - bellCp1y} ${throatX},${cy + throatR} ` +
    `L ${chamberEndX},${cy + chamberR} ` +
    `L ${injectX},${cy + chamberR} Z`;

  // Chamber zone (up to throat x)
  const chamberZone =
    `M ${injectX},${cy - chamberR} ` +
    `L ${chamberEndX},${cy - chamberR} ` +
    `L ${throatX},${cy - throatR} ` +
    `L ${throatX},${cy + throatR} ` +
    `L ${chamberEndX},${cy + chamberR} ` +
    `L ${injectX},${cy + chamberR} Z`;

  // Dimension values from result or config
  const dt = result?.dt_mm ?? null;
  const de = result?.de_mm ?? null;
  const len = result?.length_mm ?? null;
  const er_val = result?.expansion_ratio ?? config.expansion_ratio;
  const isp = result?.isp_vac ?? null;

  function label(v: number | null, unit: string, decs = 0) {
    if (v == null) return "—";
    return `${v.toFixed(decs)} ${unit}`;
  }

  // Annotation helpers
  const dimLineY = cy + exitR + 28;
  const annotY = H - 8;

  return (
    <div className="schematic-wrapper">
      <div className="schematic-title">Engine Cross-Section — {config.engine_name}</div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        style={{ maxHeight: 280, display: "block" }}
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          {/* Chamber hot gas gradient */}
          <linearGradient id="sg-chamber-fill" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#1a0a02" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#2a0f06" stopOpacity="0.9" />
          </linearGradient>
          {/* Divergent cool gradient */}
          <linearGradient id="sg-div-fill" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#2a0f06" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#041526" stopOpacity="0.9" />
          </linearGradient>
          {/* Throat glow radial */}
          <radialGradient id="sg-throat-glow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#ff6020" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#ff6020" stopOpacity="0" />
          </radialGradient>
          {/* Wall gradient */}
          <linearGradient id="sg-wall-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#1a3a5c" />
            <stop offset="100%" stopColor="#0e2040" />
          </linearGradient>
          {/* Axis dashes */}
          <pattern id="sg-grid" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
            <line x1="40" y1="0" x2="40" y2="40" stroke="#0e1e36" strokeWidth="0.5" />
            <line x1="0" y1="40" x2="40" y2="40" stroke="#0e1e36" strokeWidth="0.5" />
          </pattern>
        </defs>

        {/* Background */}
        <rect width={W} height={H} fill="var(--bg-base)" />
        <rect width={W} height={H} fill="url(#sg-grid)" />

        {/* Centreline */}
        <line
          x1={injectX - 10} y1={cy} x2={exitX + 20} y2={cy}
          stroke="#1a3060" strokeWidth="1" strokeDasharray="6 8"
        />

        {/* Interior chamber fill (hot) */}
        <path d={chamberZone} fill="url(#sg-chamber-fill)" />

        {/* Interior divergent fill (cooling blue) */}
        <path d={interiorPath} fill="url(#sg-div-fill)" />
        {/* Re-draw chamber over to get correct overlap */}
        <path d={chamberZone} fill="url(#sg-chamber-fill)" />

        {/* Throat glow */}
        <ellipse
          cx={throatX} cy={cy}
          rx={28} ry={throatR + 14}
          fill="url(#sg-throat-glow)"
        />

        {/* Combustion chamber label box */}
        <rect
          x={injectX + 8} y={cy - chamberR + 6}
          width={chamberEndX - injectX - 16} height={14}
          rx={1} fill="rgba(255,80,20,0.07)"
        />
        <text
          x={(injectX + chamberEndX) / 2} y={cy - chamberR + 16}
          textAnchor="middle"
          fill="#ff6020" fillOpacity="0.7"
          fontSize="9" fontFamily="var(--font-mono)" letterSpacing="0.12em"
        >
          COMBUSTION CHAMBER
        </text>

        {/* Wall outlines (top & bottom) */}
        <path d={upperWall} fill="none" stroke="url(#sg-wall-grad)" strokeWidth="3" strokeLinejoin="round" />
        <path d={lowerWall} fill="none" stroke="url(#sg-wall-grad)" strokeWidth="3" strokeLinejoin="round" />

        {/* Throat highlight */}
        <line
          x1={throatX} y1={cy - throatR - 2}
          x2={throatX} y2={cy + throatR + 2}
          stroke="#ff6020" strokeWidth="1.5" strokeOpacity="0.7"
        />

        {/* Injector face */}
        <line
          x1={injectX} y1={cy - chamberR - 2}
          x2={injectX} y2={cy + chamberR + 2}
          stroke="var(--border-active)" strokeWidth="3"
        />
        {/* Injector arrows */}
        {[-1, 0, 1].map((i) => (
          <g key={i} transform={`translate(${injectX - 18}, ${cy + i * (chamberR / 2.5)})`}>
            <line x1="0" y1="0" x2="14" y2="0" stroke="var(--accent-bright)" strokeWidth="1.2" strokeOpacity="0.6" />
            <polygon points="14,0 10,-3 10,3" fill="var(--accent-bright)" fillOpacity="0.6" />
          </g>
        ))}

        {/* Exit plane */}
        <line
          x1={exitX} y1={cy - exitR - 2}
          x2={exitX} y2={cy + exitR + 2}
          stroke="var(--border-active)" strokeWidth="2" strokeDasharray="3 3"
        />

        {/* ── Dimension lines ── */}
        {/* Throat diameter */}
        <g opacity="0.75">
          <line x1={throatX} y1={cy - throatR} x2={throatX} y2={cy - throatR - 18}
            stroke="var(--text-muted)" strokeWidth="0.8" />
          <line x1={throatX} y1={cy + throatR} x2={throatX} y2={cy + throatR + 18}
            stroke="var(--text-muted)" strokeWidth="0.8" />
          <line x1={throatX - 12} y1={cy - throatR - 18} x2={throatX + 12} y2={cy - throatR - 18}
            stroke="var(--text-muted)" strokeWidth="0.8" />
          <text x={throatX + 14} y={cy - throatR - 12}
            fill="var(--text-secondary)" fontSize="10" fontFamily="var(--font-mono)">
            Dt {label(dt, "mm", 1)}
          </text>
        </g>

        {/* Exit diameter */}
        <g opacity="0.75">
          <line x1={exitX + 8} y1={cy - exitR} x2={exitX + 32} y2={cy - exitR}
            stroke="var(--text-muted)" strokeWidth="0.8" />
          <line x1={exitX + 8} y1={cy + exitR} x2={exitX + 32} y2={cy + exitR}
            stroke="var(--text-muted)" strokeWidth="0.8" />
          <line x1={exitX + 32} y1={cy - exitR} x2={exitX + 32} y2={cy + exitR}
            stroke="var(--text-muted)" strokeWidth="0.8" />
          <text x={exitX + 36} y={cy - exitR / 2}
            fill="var(--text-secondary)" fontSize="10" fontFamily="var(--font-mono)">
            De {label(de, "mm", 1)}
          </text>
          <text x={exitX + 36} y={cy - exitR / 2 + 13}
            fill="var(--text-secondary)" fontSize="10" fontFamily="var(--font-mono)">
            ε = {er_val?.toFixed(2) ?? "—"}
          </text>
        </g>

        {/* Overall length annotation */}
        <g opacity="0.60">
          <line x1={injectX} y1={cy + chamberR + 22} x2={exitX} y2={cy + chamberR + 22}
            stroke="var(--text-muted)" strokeWidth="0.8" />
          <line x1={injectX} y1={cy + chamberR + 18} x2={injectX} y2={cy + chamberR + 26}
            stroke="var(--text-muted)" strokeWidth="0.8" />
          <line x1={exitX} y1={cy + chamberR + 18} x2={exitX} y2={cy + chamberR + 26}
            stroke="var(--text-muted)" strokeWidth="0.8" />
          <text
            x={(injectX + exitX) / 2} y={cy + chamberR + 36}
            textAnchor="middle"
            fill="var(--text-secondary)" fontSize="10" fontFamily="var(--font-mono)"
          >
            L = {label(len, "mm", 0)}
          </text>
        </g>

        {/* Zone labels */}
        <text
          x={throatX + 22} y={cy - throatR - 24}
          fill="#ff6020" fillOpacity="0.6"
          fontSize="8.5" fontFamily="var(--font-mono)" letterSpacing="0.1em"
        >
          THROAT
        </text>
        <text
          x={(throatX + exitX) / 2} y={cy - exitR / 2 - 2}
          textAnchor="middle"
          fill="var(--text-muted)"
          fontSize="8.5" fontFamily="var(--font-mono)" letterSpacing="0.1em"
        >
          DIVERGENT NOZZLE
        </text>

        {/* Propellant labels */}
        <text
          x={injectX - 22} y={cy - chamberR - 6}
          textAnchor="middle"
          fill="var(--text-muted)"
          fontSize="9" fontFamily="var(--font-mono)"
        >
          {config.oxidizer}
        </text>
        <text
          x={injectX - 22} y={cy + chamberR + 14}
          textAnchor="middle"
          fill="var(--text-muted)"
          fontSize="9" fontFamily="var(--font-mono)"
        >
          {config.fuel}
        </text>

        {/* Isp callout (if available) */}
        {isp != null && (
          <g>
            <rect
              x={exitX + 52} y={cy - 18}
              width={88} height={36}
              rx={2}
              fill="var(--bg-elevated)"
              stroke="var(--border-active)"
              strokeWidth="1"
            />
            <text
              x={exitX + 96} y={cy - 5}
              textAnchor="middle"
              fill="var(--text-secondary)" fontSize="8.5" fontFamily="var(--font-mono)"
              letterSpacing="0.08em"
            >
              ISP VAC
            </text>
            <text
              x={exitX + 96} y={cy + 12}
              textAnchor="middle"
              fill="var(--accent-bright)" fontSize="14" fontFamily="var(--font-mono)"
              fontWeight="500"
            >
              {isp.toFixed(1)} s
            </text>
          </g>
        )}

        {/* Nozzle type tag */}
        <text
          x={injectX + 6} y={annotY}
          fill="var(--text-muted)"
          fontSize="9" fontFamily="var(--font-mono)" letterSpacing="0.08em"
        >
          {config.nozzle_type?.toUpperCase() ?? "BELL"} NOZZLE  ·  {config.fuel} / {config.oxidizer}
          {result ? `  ·  MR ${result.combustion?.mr?.toFixed(2) ?? config.mr}` : ""}
        </text>
      </svg>

      {/* Legend row */}
      <div style={{ display: "flex", gap: 20, alignSelf: "flex-end", paddingRight: 8 }}>
        {[
          { color: "#ff6020", label: "Combustion zone" },
          { color: "var(--accent-bright)", label: "Injector" },
          { color: "var(--border-active)", label: "Exit plane" },
        ].map(({ color, label }) => (
          <div
            key={label}
            style={{
              display: "flex", alignItems: "center", gap: 5,
              fontSize: 10, color: "var(--text-muted)",
            }}
          >
            <div style={{ width: 10, height: 2, background: color, borderRadius: 1 }} />
            {label}
          </div>
        ))}
      </div>
    </div>
  );
}
