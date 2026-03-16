/**
 * API client for torch igniter design.
 */
import { useMutation } from "@tanstack/react-query";
import client from "./client";
import type { IgniterConfig, IgniterResponse } from "../types/igniter";

export async function designIgniter(config: IgniterConfig): Promise<IgniterResponse> {
  const { data } = await client.post<IgniterResponse>("/igniter/design", config);
  return data;
}

export function useIgniterMutation() {
  return useMutation({
    mutationFn: (config: IgniterConfig) => designIgniter(config),
  });
}
