import { EngineConfigForm } from "../../components/forms/EngineConfigForm";

interface ConfigurationTabProps {
  onRunDesign: () => void;
  isRunning: boolean;
}

export function ConfigurationTab({ onRunDesign, isRunning }: ConfigurationTabProps) {
  return (
    <div style={{ padding: "16px 0" }}>
      <EngineConfigForm onRunDesign={onRunDesign} isRunning={isRunning} />
    </div>
  );
}
