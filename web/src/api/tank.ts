/**
 * API client for tank simulation.
 */
import { useMutation } from "@tanstack/react-query";
import client from "./client";
import type { TankSimConfig, TankSimResponse } from "../types/tank";

export async function simulateTank(config: TankSimConfig): Promise<TankSimResponse> {
  const { data } = await client.post<TankSimResponse>("/tank/simulate", config);
  return data;
}

export function useTankMutation() {
  return useMutation({
    mutationFn: (config: TankSimConfig) => simulateTank(config),
  });
}
