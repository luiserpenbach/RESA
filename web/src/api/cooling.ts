/**
 * API client for cooling design and analysis.
 */
import { useMutation } from "@tanstack/react-query";
import client from "./client";
import type { CoolingChannelConfig, CoolingChannelResponse, CoolingAnalysisResponse } from "../types/cooling";

export async function designChannels(
  sessionId: string,
  config: CoolingChannelConfig
): Promise<CoolingChannelResponse> {
  const { data } = await client.post<CoolingChannelResponse>(
    `/cooling/channels?session_id=${sessionId}`,
    config
  );
  return data;
}

export async function analyzeCooling(
  sessionId: string
): Promise<CoolingAnalysisResponse> {
  const { data } = await client.post<CoolingAnalysisResponse>(
    `/cooling/analyze?session_id=${sessionId}`
  );
  return data;
}

export function useDesignChannelsMutation() {
  return useMutation({
    mutationFn: (params: { sessionId: string; config: CoolingChannelConfig }) =>
      designChannels(params.sessionId, params.config),
  });
}

export function useAnalyzeCoolingMutation() {
  return useMutation({
    mutationFn: (sessionId: string) => analyzeCooling(sessionId),
  });
}
