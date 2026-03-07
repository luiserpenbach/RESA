import { Button, Card, H5 } from "@blueprintjs/core";
import { exportYaml } from "../../api/engine";
import { useEngineStore } from "../../store/engineStore";
import type { EngineDesignResponse } from "../../types/engine";

interface ExportTabProps {
  result: EngineDesignResponse | null;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function ExportTab({ result }: ExportTabProps) {
  const { activeConfig } = useEngineStore();

  async function handleYamlDownload() {
    const blob = await exportYaml(activeConfig);
    const name = `${activeConfig.engine_name.replace(/\s+/g, "_")}_config.yaml`;
    downloadBlob(blob, name);
  }

  function handleContourCsvDownload() {
    if (!result?.contour_x_mm || !result.contour_y_mm) return;
    const rows = result.contour_x_mm.map(
      (x, i) => `${x},${result.contour_y_mm![i]}`
    );
    const csv = "x_mm,r_mm\n" + rows.join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    downloadBlob(blob, `${activeConfig.engine_name}_contour.csv`);
  }

  return (
    <div style={{ padding: "16px 0", maxWidth: 600 }}>
      <H5 style={{ color: "#7ba7cc", marginBottom: 12 }}>Export Options</H5>

      <Card
        style={{
          background: "#0d2137",
          border: "1px solid #1e3a5f",
          marginBottom: 12,
        }}
      >
        <H5 style={{ color: "#c9d8e8" }}>Engine Configuration (YAML)</H5>
        <p style={{ color: "#7ba7cc", marginBottom: 12 }}>
          Download the current engine configuration as a YAML file compatible
          with <code>EngineConfig.from_yaml()</code>.
        </p>
        <Button icon="download" onClick={handleYamlDownload}>
          Download YAML Config
        </Button>
      </Card>

      <Card
        style={{
          background: "#0d2137",
          border: "1px solid #1e3a5f",
        }}
      >
        <H5 style={{ color: "#c9d8e8" }}>Nozzle Contour (CSV)</H5>
        <p style={{ color: "#7ba7cc", marginBottom: 12 }}>
          Download nozzle contour coordinates (x, r in mm) as a CSV file.
        </p>
        <Button
          icon="download"
          onClick={handleContourCsvDownload}
          disabled={!result?.contour_x_mm}
        >
          Download Contour CSV
        </Button>
      </Card>
    </div>
  );
}
