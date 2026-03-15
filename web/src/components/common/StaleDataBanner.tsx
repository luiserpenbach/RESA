/**
 * Banner shown when upstream data changed but this module hasn't re-run.
 */
import { Icon } from "@blueprintjs/core";
import type { ModuleName, ModuleStatus } from "../../types/session";

interface StaleDataBannerProps {
  moduleName: ModuleName;
  moduleStatus: Record<ModuleName, ModuleStatus>;
}

export function StaleDataBanner({ moduleName, moduleStatus }: StaleDataBannerProps) {
  if (moduleStatus[moduleName] !== "stale") {
    return null;
  }

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 12px",
        background: "#331800",
        borderBottom: "1px solid #e65100",
        fontSize: 11,
        color: "#ffab40",
      }}
    >
      <Icon icon="warning-sign" size={12} style={{ color: "#e65100" }} />
      <span>
        Upstream data has changed. Re-run this analysis to update results.
      </span>
    </div>
  );
}
