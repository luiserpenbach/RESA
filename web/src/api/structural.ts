/**
 * API client for structural / wall thickness analysis.
 */
import { useMutation } from "@tanstack/react-query";
import client from "./client";
import type { WallThicknessConfig, WallThicknessResponse, MaterialListResponse } from "../types/structural";

export async function analyzeWallThickness(
  sessionId: string,
  config: WallThicknessConfig
): Promise<WallThicknessResponse> {
  const { data } = await client.post<WallThicknessResponse>(
    `/structural/wall-thickness?session_id=${sessionId}`,
    config
  );
  return data;
}

export async function listMaterials(): Promise<MaterialListResponse> {
  const { data } = await client.get<MaterialListResponse>("/structural/materials");
  return data;
}

export function useWallThicknessMutation() {
  return useMutation({
    mutationFn: (params: { sessionId: string; config: WallThicknessConfig }) =>
      analyzeWallThickness(params.sessionId, params.config),
  });
}
