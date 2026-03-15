/**
 * Types for design session management and multi-module workflow.
 */

export type ModuleName =
  | "engine"
  | "contour"
  | "cooling_channels"
  | "cooling"
  | "wall_thickness"
  | "performance"
  | "feed_system";

export type ModuleStatus = "locked" | "ready" | "completed" | "stale";

export interface SessionCreateResponse {
  session_id: string;
  module_status: Record<string, string>;
  engine_result: Record<string, unknown> | null;
}

export interface SessionStatusResponse {
  session_id: string;
  module_status: Record<string, string>;
}
