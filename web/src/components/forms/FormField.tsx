import { FormGroup } from "@blueprintjs/core";
import type { ReactNode } from "react";

interface FormFieldProps {
  label: string;
  unit?: string;
  help?: string;
  children: ReactNode;
}

/**
 * Blueprint FormGroup wrapper with optional unit label.
 */
export function FormField({ label, unit, help, children }: FormFieldProps) {
  return (
    <FormGroup
      label={
        <span>
          {label}
          {unit && (
            <span style={{ color: "#7ba7cc", fontSize: "11px", marginLeft: 4 }}>
              [{unit}]
            </span>
          )}
        </span>
      }
      helperText={help}
      style={{ marginBottom: 10 }}
    >
      {children}
    </FormGroup>
  );
}
