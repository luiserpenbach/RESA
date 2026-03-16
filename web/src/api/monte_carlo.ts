/**
 * API client for Monte Carlo uncertainty analysis.
 */
import { useMutation } from "@tanstack/react-query";
import client from "./client";
import type { MonteCarloConfig, MonteCarloResponse } from "../types/monte_carlo";

export async function runMonteCarlo(
  sessionId: string,
  config: MonteCarloConfig
): Promise<MonteCarloResponse> {
  const { data } = await client.post<MonteCarloResponse>(
    `/monte-carlo/run?session_id=${sessionId}`,
    config
  );
  return data;
}

export function useMonteCarloMutation() {
  return useMutation({
    mutationFn: (params: { sessionId: string; config: MonteCarloConfig }) =>
      runMonteCarlo(params.sessionId, params.config),
  });
}
