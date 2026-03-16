/**
 * API client for swirl injector design.
 */
import { useMutation } from "@tanstack/react-query";
import client from "./client";
import type { InjectorConfig, InjectorResponse } from "../types/injector";

export async function designInjector(config: InjectorConfig): Promise<InjectorResponse> {
  const { data } = await client.post<InjectorResponse>("/injector/design", config);
  return data;
}

export function useInjectorMutation() {
  return useMutation({
    mutationFn: (config: InjectorConfig) => designInjector(config),
  });
}
