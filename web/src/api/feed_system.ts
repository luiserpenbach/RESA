/**
 * API client for feed system analysis.
 */
import { useMutation } from "@tanstack/react-query";
import client from "./client";
import type { FeedSystemConfig, FeedSystemResponse } from "../types/feed_system";

export async function analyzeFeedSystem(
  sessionId: string,
  config: FeedSystemConfig
): Promise<FeedSystemResponse> {
  const { data } = await client.post<FeedSystemResponse>(
    `/feed-system/analyze?session_id=${sessionId}`,
    config
  );
  return data;
}

export function useFeedSystemMutation() {
  return useMutation({
    mutationFn: (params: { sessionId: string; config: FeedSystemConfig }) =>
      analyzeFeedSystem(params.sessionId, params.config),
  });
}
