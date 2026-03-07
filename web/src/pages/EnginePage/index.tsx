import { useState } from "react";
import { Tab, Tabs } from "@blueprintjs/core";
import { ConfigurationTab } from "./ConfigurationTab";
import { ResultsTab } from "./ResultsTab";
import { ReportTab } from "./ReportTab";
import { ExportTab } from "./ExportTab";
import { useDesignMutation } from "../../api/engine";
import { useEngineStore } from "../../store/engineStore";

export default function EnginePage() {
  const [activeTab, setActiveTab] = useState<string>("config");
  const { activeConfig, lastDesignResult, setResult } = useEngineStore();
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const designMutation = useDesignMutation();

  async function handleRunDesign() {
    setErrorMsg(null);
    try {
      const result = await designMutation.mutateAsync({
        config: activeConfig,
        withCooling: false,
      });
      setResult(result);
      setActiveTab("results");
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "Design failed. Check configuration.";
      setErrorMsg(msg);
      setActiveTab("results");
    }
  }

  return (
    <div>
      <div style={{ marginBottom: 20 }}>
        <h2
          style={{
            color: "#e8f4fd",
            margin: 0,
            fontSize: "22px",
            fontWeight: 700,
          }}
        >
          Engine Design
        </h2>
        <p style={{ color: "#7ba7cc", margin: "4px 0 0" }}>
          Configure and run thrust chamber design analysis
        </p>
      </div>

      <Tabs
        id="engine-tabs"
        selectedTabId={activeTab}
        onChange={(t) => setActiveTab(String(t))}
        renderActiveTabPanelOnly={false}
      >
        <Tab
          id="config"
          title="Configuration"
          panel={
            <ConfigurationTab
              onRunDesign={handleRunDesign}
              isRunning={designMutation.isPending}
            />
          }
        />
        <Tab
          id="results"
          title="Results"
          panel={
            <ResultsTab
              result={lastDesignResult}
              isLoading={designMutation.isPending}
              error={errorMsg}
            />
          }
        />
        <Tab
          id="report"
          title="Report"
          panel={
            <ReportTab
              result={lastDesignResult}
            />
          }
        />
        <Tab
          id="export"
          title="Export"
          panel={
            <ExportTab
              result={lastDesignResult}
            />
          }
        />
      </Tabs>
    </div>
  );
}
