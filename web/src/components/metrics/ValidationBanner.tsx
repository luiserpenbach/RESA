import { Callout } from "@blueprintjs/core";
import type { ValidationResponse } from "../../types/engine";

interface ValidationBannerProps {
  validation: ValidationResponse | null;
}

export function ValidationBanner({ validation }: ValidationBannerProps) {
  if (!validation) return null;

  return (
    <div style={{ marginBottom: 12 }}>
      {!validation.is_valid && validation.errors.length > 0 && (
        <Callout intent="danger" title="Configuration Errors" icon="error">
          <ul style={{ margin: "4px 0", paddingLeft: 20 }}>
            {validation.errors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </Callout>
      )}
      {validation.warnings.length > 0 && (
        <Callout
          intent="warning"
          title="Warnings"
          icon="warning-sign"
          style={{ marginTop: validation.errors.length > 0 ? 8 : 0 }}
        >
          <ul style={{ margin: "4px 0", paddingLeft: 20 }}>
            {validation.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </Callout>
      )}
      {validation.is_valid && validation.warnings.length === 0 && (
        <Callout intent="success" icon="tick">
          Configuration is valid.
        </Callout>
      )}
    </div>
  );
}
