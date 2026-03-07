import { Spinner, OverlayToaster } from "@blueprintjs/core";

interface LoadingOverlayProps {
  label?: string;
}

export function LoadingOverlay({ label = "Running..." }: LoadingOverlayProps) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 12,
        padding: 32,
      }}
    >
      <Spinner size={40} intent="primary" />
      <span style={{ color: "#7ba7cc" }}>{label}</span>
    </div>
  );
}
