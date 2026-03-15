/**
 * API client for nozzle contour design.
 */
import { useMutation } from "@tanstack/react-query";
import client from "./client";
import type { ContourConfig, ContourResponse } from "../types/contour";

export async function generateContour(
  sessionId: string,
  config: ContourConfig
): Promise<ContourResponse> {
  const { data } = await client.post<ContourResponse>(
    `/contour/generate?session_id=${sessionId}`,
    config
  );
  return data;
}

export async function exportContour(
  sessionId: string,
  format: string = "csv"
): Promise<Blob> {
  const { data } = await client.get(`/contour/export`, {
    params: { session_id: sessionId, format },
    responseType: "blob",
  });
  return data as Blob;
}

export function useGenerateContourMutation() {
  return useMutation({
    mutationFn: (params: { sessionId: string; config: ContourConfig }) =>
      generateContour(params.sessionId, params.config),
  });
}
