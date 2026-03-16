import { useState } from "react";
import {
  Button,
  NumericInput,
  HTMLSelect,
  Switch,
  Slider,
  Tooltip,
} from "@blueprintjs/core";
import type { ReactNode } from "react";
import { PARAM_DOCS } from "../../data/paramDocs";

import { ValidationBanner } from "../metrics/ValidationBanner";
import { useValidateMutation } from "../../api/engine";
import { useEngineStore } from "../../store/engineStore";
import type { EngineConfigRequest, ValidationResponse } from "../../types/engine";

const FUELS = [
  "Ethanol90",
  "Ethanol80",
  "Ethanol",
  "IPA",
  "Methane",
  "LCH4",
  "RP1",
  "Kerosene",
  "Propane",
];

const OXIDIZERS = ["N2O", "Nitrous", "LOX", "O2", "H2O2"];
const NOZZLE_TYPES = ["bell", "conical", "ideal"];

interface EngineConfigFormProps {
  onRunDesign?: () => void;
  isRunning?: boolean;
  compact?: boolean;
}

/* ── Shared table styles ────────────────────────────────────────── */
const inputStyle: React.CSSProperties = {
  width: "100%",
  height: 26,
  padding: "0 7px",
  background: "var(--bg-base)",
  border: "1px solid var(--border-default)",
  borderRadius: "var(--radius-sm)",
  color: "var(--text-primary)",
  fontFamily: "var(--font-mono)",
  fontSize: 12,
  boxSizing: "border-box",
};

function ParamRow({
  label,
  paramKey,
  children,
}: {
  label: string;
  paramKey?: string;
  children: ReactNode;
}) {
  const doc = paramKey ? PARAM_DOCS[paramKey] : undefined;

  return (
    <tr>
      <td className="param-label">
        <span className="param-label-text">{label}</span>
        {doc && (
          <Tooltip
            content={
              <div className="param-tooltip-content">
                <div className="param-tooltip-desc">{doc.description}</div>
                {doc.range && (
                  <div className="param-tooltip-range">
                    <span className="param-tooltip-tag">Range</span>
                    {doc.range}
                  </div>
                )}
                {doc.note && (
                  <div className="param-tooltip-note">{doc.note}</div>
                )}
              </div>
            }
            placement="right"
            compact
          >
            <span className="param-help-icon" tabIndex={0}>ⓘ</span>
          </Tooltip>
        )}
      </td>
      <td className="param-value">{children}</td>
    </tr>
  );
}

function SectionHeader({ children }: { children: ReactNode }) {
  return (
    <div className="param-section-header">{children}</div>
  );
}

export function EngineConfigForm({
  onRunDesign,
  isRunning = false,
}: EngineConfigFormProps) {
  const { activeConfig, setConfig } = useEngineStore();
  const [localConfig, setLocalConfig] = useState<EngineConfigRequest>(activeConfig);
  const [validation, setValidation] = useState<ValidationResponse | null>(null);

  const validateMutation = useValidateMutation();

  function updateField<K extends keyof EngineConfigRequest>(
    key: K,
    value: EngineConfigRequest[K]
  ) {
    setLocalConfig((prev) => {
      const next = { ...prev, [key]: value };
      setConfig(next);
      return next;
    });
  }

  async function handleValidate() {
    setConfig(localConfig);
    const result = await validateMutation.mutateAsync(localConfig);
    setValidation(result);
  }

  function handleRunDesign() {
    setConfig(localConfig);
    onRunDesign?.();
  }

  return (
    <div>
      {/* ── IDENTIFICATION ─────────────────────────────── */}
      <SectionHeader>Identification</SectionHeader>
      <table className="param-table">
        <tbody>
          <ParamRow label="Engine Name">
            <input
              style={inputStyle}
              type="text"
              value={localConfig.engine_name}
              onChange={(e) => updateField("engine_name", e.target.value)}
            />
          </ParamRow>
          <ParamRow label="Version">
            <input
              style={inputStyle}
              type="text"
              value={localConfig.version}
              onChange={(e) => updateField("version", e.target.value)}
            />
          </ParamRow>
          <ParamRow label="Designer">
            <input
              style={inputStyle}
              type="text"
              value={localConfig.designer}
              onChange={(e) => updateField("designer", e.target.value)}
            />
          </ParamRow>
        </tbody>
      </table>

      {/* ── PROPELLANTS ────────────────────────────────── */}
      <SectionHeader>Propellants</SectionHeader>
      <table className="param-table">
        <tbody>
          <ParamRow label="Fuel">
            <select
              className="form-select"
              value={localConfig.fuel}
              onChange={(e) => updateField("fuel", e.target.value)}
            >
              {FUELS.map((f) => <option key={f} value={f}>{f}</option>)}
            </select>
          </ParamRow>
          <ParamRow label="Oxidizer">
            <select
              className="form-select"
              value={localConfig.oxidizer}
              onChange={(e) => updateField("oxidizer", e.target.value)}
            >
              {OXIDIZERS.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </ParamRow>
          <ParamRow label="Fuel Inj. Temp [K]" paramKey="fuel_injection_temp_k">
            <NumericInput
              value={localConfig.fuel_injection_temp_k}
              onValueChange={(v) => updateField("fuel_injection_temp_k", v)}
              min={100} max={500} stepSize={1} fill
            />
          </ParamRow>
          <ParamRow label="Ox. Inj. Temp [K]" paramKey="oxidizer_injection_temp_k">
            <NumericInput
              value={localConfig.oxidizer_injection_temp_k}
              onValueChange={(v) => updateField("oxidizer_injection_temp_k", v)}
              min={100} max={500} stepSize={1} fill
            />
          </ParamRow>
        </tbody>
      </table>

      {/* ── PERFORMANCE TARGETS ────────────────────────── */}
      <SectionHeader>Performance Targets</SectionHeader>
      <table className="param-table">
        <tbody>
          <ParamRow label="Thrust [N]" paramKey="thrust_n">
            <NumericInput
              value={localConfig.thrust_n}
              onValueChange={(v) => updateField("thrust_n", v)}
              min={1} stepSize={100} fill
            />
          </ParamRow>
          <ParamRow label="Chamber Pressure [bar]" paramKey="pc_bar">
            <NumericInput
              value={localConfig.pc_bar}
              onValueChange={(v) => updateField("pc_bar", v)}
              min={1} max={300} stepSize={1} fill
            />
          </ParamRow>
          <ParamRow label="Mixture Ratio [O/F]" paramKey="mr">
            <NumericInput
              value={localConfig.mr}
              onValueChange={(v) => updateField("mr", v)}
              min={0.5} max={15} stepSize={0.1} minorStepSize={0.01} fill
            />
          </ParamRow>
          <ParamRow label="Combustion Efficiency" paramKey="eff_combustion">
            <NumericInput
              value={localConfig.eff_combustion}
              onValueChange={(v) => updateField("eff_combustion", v)}
              min={0.5} max={1.0} stepSize={0.01} minorStepSize={0.001} fill
            />
          </ParamRow>
          <ParamRow label="Nozzle Div. Efficiency" paramKey="eff_nozzle_divergence">
            <NumericInput
              value={localConfig.eff_nozzle_divergence}
              onValueChange={(v) => updateField("eff_nozzle_divergence", v)}
              min={0.8} max={1.0} stepSize={0.001} minorStepSize={0.001} fill
            />
          </ParamRow>
          <ParamRow label="Freeze at Throat" paramKey="freeze_at_throat">
            <Switch
              checked={localConfig.freeze_at_throat}
              onChange={(e) => updateField("freeze_at_throat", e.target.checked)}
              style={{ marginBottom: 0 }}
            />
          </ParamRow>
        </tbody>
      </table>

      {/* ── NOZZLE DESIGN ──────────────────────────────── */}
      <SectionHeader>Nozzle Design</SectionHeader>
      <table className="param-table">
        <tbody>
          <ParamRow label="Nozzle Type">
            <HTMLSelect
              value={localConfig.nozzle_type}
              onChange={(e) =>
                updateField("nozzle_type", e.target.value as EngineConfigRequest["nozzle_type"])
              }
              options={NOZZLE_TYPES}
              fill
            />
          </ParamRow>
          <ParamRow label="Expansion Ratio" paramKey="expansion_ratio">
            <NumericInput
              value={localConfig.expansion_ratio}
              onValueChange={(v) => updateField("expansion_ratio", v)}
              min={0} stepSize={0.1} fill
            />
          </ParamRow>
          <ParamRow label="Exit Pressure [bar]" paramKey="p_exit_bar">
            <NumericInput
              value={localConfig.p_exit_bar}
              onValueChange={(v) => updateField("p_exit_bar", v)}
              min={0.01} stepSize={0.1} fill
            />
          </ParamRow>
          <ParamRow label="L* [mm]" paramKey="L_star">
            <NumericInput
              value={localConfig.L_star}
              onValueChange={(v) => updateField("L_star", v)}
              min={200} max={3000} stepSize={50} fill
            />
          </ParamRow>
          <ParamRow label="Contraction Ratio" paramKey="contraction_ratio">
            <NumericInput
              value={localConfig.contraction_ratio}
              onValueChange={(v) => updateField("contraction_ratio", v)}
              min={2} max={20} stepSize={0.5} fill
            />
          </ParamRow>
          <ParamRow label="Convergent Angle [deg]" paramKey="theta_convergent">
            <NumericInput
              value={localConfig.theta_convergent}
              onValueChange={(v) => updateField("theta_convergent", v)}
              min={15} max={60} stepSize={1} fill
            />
          </ParamRow>
          <ParamRow label="Exit Half-Angle [deg]" paramKey="theta_exit">
            <NumericInput
              value={localConfig.theta_exit}
              onValueChange={(v) => updateField("theta_exit", v)}
              min={5} max={30} stepSize={1} fill
            />
          </ParamRow>
          <ParamRow label="Bell Fraction" paramKey="bell_fraction">
            <div style={{ padding: "6px 0 2px" }}>
              <Slider
                value={localConfig.bell_fraction}
                onChange={(v) => updateField("bell_fraction", v)}
                min={0.6} max={1.0} stepSize={0.01} labelStepSize={0.2}
                labelRenderer={(v) => v.toFixed(2)}
              />
            </div>
          </ParamRow>
        </tbody>
      </table>

      {/* ── VALIDATION / RUN ──────────────────────────── */}
      <div style={{ marginTop: 16 }}>
        <ValidationBanner validation={validation} />
        <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
          <Button
            icon="endorsed"
            onClick={handleValidate}
            loading={validateMutation.isPending}
            disabled={isRunning}
          >
            Validate
          </Button>
          <Button
            intent="primary"
            icon="play"
            onClick={handleRunDesign}
            loading={isRunning}
            disabled={validateMutation.isPending}
          >
            Run Design
          </Button>
        </div>
      </div>
    </div>
  );
}
