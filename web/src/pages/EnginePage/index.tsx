import { useState } from "react";
import { TopBar } from "../../components/layout/TopBar";
import { StatusBar } from "../../components/layout/StatusBar";
import { WorkspacePanel } from "../../components/workspace/WorkspacePanel";
import { MetricsPanel } from "../../components/metrics/MetricsPanel";
import { CommandPalette } from "../../components/ui/CommandPalette";
import { EngineConfigForm } from "../../components/forms/EngineConfigForm";
import { useDesignMutation, useValidateMutation, exportYaml } from "../../api/engine";
import { useEngineStore } from "../../store/engineStore";
import { useUiStore } from "../../store/uiStore";

export default function EnginePage() {
  const { activeConfig, lastDesignResult, setResult } = useEngineStore();
  const { setLastRunTime, setLastRunDuration, setWorkspaceTab } = useUiStore();
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const designMutation = useDesignMutation();
  const validateMutation = useValidateMutation();

  async function handleRunDesign() {
    setErrorMsg(null);
    const t0 = Date.now();
    try {
      const result = await designMutation.mutateAsync({
        config: activeConfig,
        withCooling: false,
      });
      setResult(result);
      setLastRunTime(Date.now());
      setLastRunDuration(Date.now() - t0);
      // Switch to dashboard if we have plots
      if (result.figure_dashboard) setWorkspaceTab("dashboard");
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "Design failed. Check configuration.";
      setErrorMsg(msg);
    }
  }

  async function handleValidate() {
    try {
      await validateMutation.mutateAsync(activeConfig);
    } catch { /* mutation state holds the error */ }
  }

  async function handleExportYaml() {
    try {
      const blob = await exportYaml(activeConfig);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${activeConfig.engine_name ?? "engine"}.yaml`;
      a.click();
      URL.revokeObjectURL(url);
    } catch { /* silent */ }
  }

  const isRunning = designMutation.isPending;

  return (
    <>
      {/* ── Top bar (spans all 3 columns via grid row 1) ── */}
      <TopBar onRunDesign={handleRunDesign} isRunning={isRunning} />

      {/* ── Left panel: Configuration ── */}
      <div className="app-left-panel">
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
            Parameters
          </span>
        </div>

        {/* Scrollable form */}
        <div className="panel-scroll" style={{ padding: "12px 16px" }}>
          <EngineConfigForm
            onRunDesign={handleRunDesign}
            isRunning={isRunning}
            compact
          />
        </div>
      </div>

      {/* ── Center: Workspace ── */}
      <div className="app-workspace">
        <WorkspacePanel
          config={activeConfig}
          result={lastDesignResult}
          isLoading={isRunning}
        />
      </div>

      {/* ── Right panel: Metrics ── */}
      <div className="app-right-panel">
        <MetricsPanel
          result={lastDesignResult}
          config={activeConfig}
          isLoading={isRunning}
        />
      </div>

      {/* ── Status bar (spans all 3 columns via grid row 3) ── */}
      <StatusBar isRunning={isRunning} error={errorMsg} />

      {/* ── Command Palette overlay ── */}
      <CommandPalette
        onRunDesign={handleRunDesign}
        onValidate={handleValidate}
        onExportYaml={handleExportYaml}
      />
    </>
  );
}
