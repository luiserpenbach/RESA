import { Callout } from "@blueprintjs/core";

interface ErrorCalloutProps {
  title?: string;
  message: string;
}

export function ErrorCallout({ title = "Error", message }: ErrorCalloutProps) {
  return (
    <Callout intent="danger" title={title} icon="error">
      {message}
    </Callout>
  );
}
