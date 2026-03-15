/**
 * API client for session management.
 */
import { useMutation } from "@tanstack/react-query";
import client from "./client";
import type { EngineConfigRequest } from "../types/engine";
import type { SessionCreateResponse, SessionStatusResponse } from "../types/session";

export async function createSession(
  config: EngineConfigRequest,
  withCooling = false
): Promise<SessionCreateResponse> {
  const { data } = await client.post<SessionCreateResponse>(
    `/session/create?with_cooling=${withCooling}`,
    config
  );
  return data;
}

export async function getSessionStatus(
  sessionId: string
): Promise<SessionStatusResponse> {
  const { data } = await client.get<SessionStatusResponse>(
    `/session/${sessionId}/status`
  );
  return data;
}

export async function deleteSession(sessionId: string): Promise<void> {
  await client.delete(`/session/${sessionId}`);
}

export function useCreateSessionMutation() {
  return useMutation({
    mutationFn: (params: { config: EngineConfigRequest; withCooling?: boolean }) =>
      createSession(params.config, params.withCooling),
  });
}
