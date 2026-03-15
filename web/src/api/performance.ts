/**
 * API client for performance maps.
 */
import { useMutation } from "@tanstack/react-query";
import client from "./client";
import type { PerformanceMapConfig, PerformanceFullResponse } from "../types/performance";

export async function getPerformanceMaps(
  sessionId: string,
  config: PerformanceMapConfig
): Promise<PerformanceFullResponse> {
  const { data } = await client.post<PerformanceFullResponse>(
    `/performance/full?session_id=${sessionId}`,
    config
  );
  return data;
}

export function usePerformanceMapsMutation() {
  return useMutation({
    mutationFn: (params: { sessionId: string; config: PerformanceMapConfig }) =>
      getPerformanceMaps(params.sessionId, params.config),
  });
}
