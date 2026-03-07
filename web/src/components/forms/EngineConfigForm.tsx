import { useState } from "react";
import {
  Button,
  Divider,
  H5,
  InputGroup,
  NumericInput,
  HTMLSelect,
  Switch,
  TextArea,
  Slider,
} from "@blueprintjs/core";
import { FormField } from "./FormField";
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

const COOLING_MODES = ["counter-flow", "co-flow"];

const WALL_MATERIALS = [
  "inconel718",
  "inconel625",
  "copper",
  "stainless316",
  "aluminum6061",
  "haynes230",
];

const COOLANTS = [
  "REFPROP::NitrousOxide",
  "REFPROP::Ethanol",
  "Water",
  "REFPROP::Propane",
];

interface EngineConfigFormProps {
  onRunDesign?: () => void;
  isRunning?: boolean;
  /** Compact mode for use inside a panel (no max-width, tighter spacing). */
  compact?: boolean;
}

export function EngineConfigForm({
  onRunDesign,
  isRunning = false,
  compact = false,
}: EngineConfigFormProps) {
  const { activeConfig, setConfig } = useEngineStore();
  const [localConfig, setLocalConfig] =
    useState<EngineConfigRequest>(activeConfig);
  const [validation, setValidation] = useState<ValidationResponse | null>(null);

  const validateMutation = useValidateMutation();

  function updateField<K extends keyof EngineConfigRequest>(
    key: K,
    value: EngineConfigRequest[K]
  ) {
    setLocalConfig((prev) => {
      const next = { ...prev, [key]: value };
      setConfig(next); // keep global store in sync so TopBar RUN DESIGN works
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
    <div style={{ maxWidth: compact ? undefined : 900 }}>
      {/* ── IDENTIFICATION ──────────────────────────────────── */}
      <H5 style={{ color: "#7ba7cc", marginBottom: 12 }}>Identification</H5>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <FormField label="Engine Name">
          <InputGroup
            value={localConfig.engine_name}
            onChange={(e) => updateField("engine_name", e.target.value)}
          />
        </FormField>
        <FormField label="Version">
          <InputGroup
            value={localConfig.version}
            onChange={(e) => updateField("version", e.target.value)}
          />
        </FormField>
        <FormField label="Designer">
          <InputGroup
            value={localConfig.designer}
            onChange={(e) => updateField("designer", e.target.value)}
          />
        </FormField>
      </div>
      <FormField label="Description">
        <TextArea
          value={localConfig.description}
          onChange={(e) => updateField("description", e.target.value)}
          rows={2}
          style={{ width: "100%", resize: "vertical" }}
        />
      </FormField>

      <Divider style={{ margin: "16px 0" }} />

      {/* ── PROPELLANTS ────────────────────────────────────── */}
      <H5 style={{ color: "#7ba7cc", marginBottom: 12 }}>Propellants</H5>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <FormField label="Fuel">
          <HTMLSelect
            value={localConfig.fuel}
            onChange={(e) => updateField("fuel", e.target.value)}
            options={FUELS}
            fill
          />
        </FormField>
        <FormField label="Oxidizer">
          <HTMLSelect
            value={localConfig.oxidizer}
            onChange={(e) => updateField("oxidizer", e.target.value)}
            options={OXIDIZERS}
            fill
          />
        </FormField>
        <FormField label="Fuel Injection Temp" unit="K">
          <NumericInput
            value={localConfig.fuel_injection_temp_k}
            onValueChange={(v) => updateField("fuel_injection_temp_k", v)}
            min={100}
            max={500}
            stepSize={1}
            fill
          />
        </FormField>
        <FormField label="Oxidizer Injection Temp" unit="K">
          <NumericInput
            value={localConfig.oxidizer_injection_temp_k}
            onValueChange={(v) => updateField("oxidizer_injection_temp_k", v)}
            min={100}
            max={500}
            stepSize={1}
            fill
          />
        </FormField>
      </div>

      <Divider style={{ margin: "16px 0" }} />

      {/* ── PERFORMANCE TARGETS ─────────────────────────────── */}
      <H5 style={{ color: "#7ba7cc", marginBottom: 12 }}>Performance Targets</H5>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <FormField label="Thrust" unit="N">
          <NumericInput
            value={localConfig.thrust_n}
            onValueChange={(v) => updateField("thrust_n", v)}
            min={1}
            stepSize={100}
            fill
          />
        </FormField>
        <FormField label="Chamber Pressure" unit="bar">
          <NumericInput
            value={localConfig.pc_bar}
            onValueChange={(v) => updateField("pc_bar", v)}
            min={1}
            max={300}
            stepSize={1}
            fill
          />
        </FormField>
        <FormField label="Mixture Ratio" unit="O/F">
          <NumericInput
            value={localConfig.mr}
            onValueChange={(v) => updateField("mr", v)}
            min={0.5}
            max={15}
            stepSize={0.1}
            minorStepSize={0.01}
            fill
          />
        </FormField>
        <FormField label="Combustion Efficiency">
          <NumericInput
            value={localConfig.eff_combustion}
            onValueChange={(v) => updateField("eff_combustion", v)}
            min={0.5}
            max={1.0}
            stepSize={0.01}
            minorStepSize={0.001}
            fill
          />
        </FormField>
        <FormField label="Nozzle Divergence Efficiency">
          <NumericInput
            value={localConfig.eff_nozzle_divergence}
            onValueChange={(v) => updateField("eff_nozzle_divergence", v)}
            min={0.8}
            max={1.0}
            stepSize={0.001}
            minorStepSize={0.001}
            fill
          />
        </FormField>
        <FormField label="Freeze at Throat">
          <Switch
            checked={localConfig.freeze_at_throat}
            onChange={(e) =>
              updateField("freeze_at_throat", e.target.checked)
            }
            label="Enable frozen equilibrium"
          />
        </FormField>
      </div>

      <Divider style={{ margin: "16px 0" }} />

      {/* ── NOZZLE DESIGN ───────────────────────────────────── */}
      <H5 style={{ color: "#7ba7cc", marginBottom: 12 }}>Nozzle Design</H5>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <FormField label="Nozzle Type">
          <HTMLSelect
            value={localConfig.nozzle_type}
            onChange={(e) =>
              updateField(
                "nozzle_type",
                e.target.value as EngineConfigRequest["nozzle_type"]
              )
            }
            options={NOZZLE_TYPES}
            fill
          />
        </FormField>
        <FormField
          label="Expansion Ratio"
          help="Set to 0 for automatic calculation"
        >
          <NumericInput
            value={localConfig.expansion_ratio}
            onValueChange={(v) => updateField("expansion_ratio", v)}
            min={0}
            stepSize={0.1}
            fill
          />
        </FormField>
        <FormField label="Exit Pressure" unit="bar">
          <NumericInput
            value={localConfig.p_exit_bar}
            onValueChange={(v) => updateField("p_exit_bar", v)}
            min={0.01}
            stepSize={0.1}
            fill
          />
        </FormField>
        <FormField label="L*" unit="mm">
          <NumericInput
            value={localConfig.L_star}
            onValueChange={(v) => updateField("L_star", v)}
            min={200}
            max={3000}
            stepSize={50}
            fill
          />
        </FormField>
        <FormField label="Contraction Ratio">
          <NumericInput
            value={localConfig.contraction_ratio}
            onValueChange={(v) => updateField("contraction_ratio", v)}
            min={2}
            max={20}
            stepSize={0.5}
            fill
          />
        </FormField>
        <FormField label="Convergent Half-Angle" unit="deg">
          <NumericInput
            value={localConfig.theta_convergent}
            onValueChange={(v) => updateField("theta_convergent", v)}
            min={15}
            max={60}
            stepSize={1}
            fill
          />
        </FormField>
        <FormField label="Exit Half-Angle" unit="deg">
          <NumericInput
            value={localConfig.theta_exit}
            onValueChange={(v) => updateField("theta_exit", v)}
            min={5}
            max={30}
            stepSize={1}
            fill
          />
        </FormField>
        <FormField label="Bell Fraction">
          <div style={{ padding: "4px 0" }}>
            <Slider
              value={localConfig.bell_fraction}
              onChange={(v) => updateField("bell_fraction", v)}
              min={0.6}
              max={1.0}
              stepSize={0.01}
              labelStepSize={0.1}
              labelRenderer={(v) => v.toFixed(2)}
            />
          </div>
        </FormField>
      </div>

      <Divider style={{ margin: "16px 0" }} />

      {/* ── COOLING SYSTEM ───────────────────────────────────── */}
      <H5 style={{ color: "#7ba7cc", marginBottom: 12 }}>Cooling System</H5>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <FormField label="Coolant">
          <HTMLSelect
            value={localConfig.coolant_name}
            onChange={(e) => updateField("coolant_name", e.target.value)}
            options={COOLANTS}
            fill
          />
        </FormField>
        <FormField label="Cooling Mode">
          <HTMLSelect
            value={localConfig.cooling_mode}
            onChange={(e) =>
              updateField(
                "cooling_mode",
                e.target.value as EngineConfigRequest["cooling_mode"]
              )
            }
            options={COOLING_MODES}
            fill
          />
        </FormField>
        <FormField label="Coolant Inlet Pressure" unit="bar">
          <NumericInput
            value={localConfig.coolant_p_in_bar}
            onValueChange={(v) => updateField("coolant_p_in_bar", v)}
            min={1}
            stepSize={1}
            fill
          />
        </FormField>
        <FormField label="Coolant Inlet Temperature" unit="K">
          <NumericInput
            value={localConfig.coolant_t_in_k}
            onValueChange={(v) => updateField("coolant_t_in_k", v)}
            min={150}
            max={600}
            stepSize={1}
            fill
          />
        </FormField>
        <FormField label="Channel Width at Throat" unit="mm">
          <NumericInput
            value={localConfig.channel_width_throat * 1000}
            onValueChange={(v) => updateField("channel_width_throat", v / 1000)}
            min={0.1}
            max={10}
            stepSize={0.1}
            minorStepSize={0.01}
            fill
          />
        </FormField>
        <FormField label="Channel Height" unit="mm">
          <NumericInput
            value={localConfig.channel_height * 1000}
            onValueChange={(v) => updateField("channel_height", v / 1000)}
            min={0.1}
            max={10}
            stepSize={0.1}
            minorStepSize={0.01}
            fill
          />
        </FormField>
        <FormField label="Rib Width at Throat" unit="mm">
          <NumericInput
            value={localConfig.rib_width_throat * 1000}
            onValueChange={(v) => updateField("rib_width_throat", v / 1000)}
            min={0.1}
            max={10}
            stepSize={0.1}
            minorStepSize={0.01}
            fill
          />
        </FormField>
        <FormField label="Wall Thickness" unit="mm">
          <NumericInput
            value={localConfig.wall_thickness * 1000}
            onValueChange={(v) => updateField("wall_thickness", v / 1000)}
            min={0.1}
            max={10}
            stepSize={0.1}
            minorStepSize={0.01}
            fill
          />
        </FormField>
        <FormField label="Wall Material">
          <HTMLSelect
            value={localConfig.wall_material}
            onChange={(e) => updateField("wall_material", e.target.value)}
            options={WALL_MATERIALS}
            fill
          />
        </FormField>
        <FormField label="Coolant Mass Fraction">
          <NumericInput
            value={localConfig.coolant_mass_fraction}
            onValueChange={(v) => updateField("coolant_mass_fraction", v)}
            min={0.1}
            max={1.0}
            stepSize={0.05}
            fill
          />
        </FormField>
      </div>

      <Divider style={{ margin: "16px 0" }} />

      {/* ── VALIDATION / RUN ─────────────────────────────────── */}
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
  );
}
