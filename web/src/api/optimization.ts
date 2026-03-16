/**
 * API client for design optimization.
 */
import { useMutation } from "@tanstack/react-query";
import client from "./client";
import type { OptimizationConfig, OptimizationResponse } from "../types/optimization";

export async function runOptimization(
  sessionId: string,
  config: OptimizationConfig
): Promise<OptimizationResponse> {
  const { data } = await client.post<OptimizationResponse>(
    `/optimization/run?session_id=${sessionId}`,
    config
  );
  return data;
}

export function useOptimizationMutation() {
  return useMutation({
    mutationFn: (params: { sessionId: string; config: OptimizationConfig }) =>
      runOptimization(params.sessionId, params.config),
  });
}
