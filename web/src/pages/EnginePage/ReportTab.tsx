import { Button, Card, Callout, H5 } from "@blueprintjs/core";
import type { EngineDesignResponse } from "../../types/engine";

interface ReportTabProps {
  result: EngineDesignResponse | null;
}

export function ReportTab({ result }: ReportTabProps) {
  if (!result) {
    return (
      <div style={{ padding: "16px 0", color: "#5c7d9e" }}>
        Run a design first to generate a report.
      </div>
    );
  }

  return (
    <div style={{ padding: "16px 0", maxWidth: 600 }}>
      <H5 style={{ color: "#7ba7cc", marginBottom: 12 }}>Report Generation</H5>
      <Card style={{ background: "#0d2137", border: "1px solid #1e3a5f" }}>
        <p style={{ color: "#c9d8e8", marginBottom: 16 }}>
          Generate a professional HTML report with embedded interactive charts.
          The report includes all performance metrics, visualizations, and
          analysis results.
        </p>
        <Callout intent="primary" icon="info-sign" style={{ marginBottom: 16 }}>
          Full report generation endpoint is available at{" "}
          <code>POST /api/v1/engine/report</code>. The HTML report can be
          downloaded once the reporting endpoint is integrated.
        </Callout>
        <Button
          icon="document"
          intent="primary"
          disabled
          title="Reporting endpoint coming soon"
        >
          Generate HTML Report
        </Button>
      </Card>
    </div>
  );
}
