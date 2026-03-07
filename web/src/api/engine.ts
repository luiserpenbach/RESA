import { useMutation } from "@tanstack/react-query";
import client from "./client";
import type {
  EngineConfigRequest,
  EngineDesignResponse,
  ValidationResponse,
} from "../types/engine";

export async function validateConfig(
  config: EngineConfigRequest
): Promise<ValidationResponse> {
  const { data } = await client.post<ValidationResponse>(
    "/engine/validate",
    config
  );
  return data;
}

export async function designEngine(
  config: EngineConfigRequest,
  withCooling = false
): Promise<EngineDesignResponse> {
  const { data } = await client.post<EngineDesignResponse>(
    `/engine/design?with_cooling=${withCooling}`,
    config
  );
  return data;
}

export function useValidateMutation() {
  return useMutation({
    mutationFn: validateConfig,
  });
}

export function useDesignMutation() {
  return useMutation({
    mutationFn: ({
      config,
      withCooling,
    }: {
      config: EngineConfigRequest;
      withCooling?: boolean;
    }) => designEngine(config, withCooling),
  });
}

export async function exportYaml(config: EngineConfigRequest): Promise<Blob> {
  const response = await client.post("/config/export-yaml", config, {
    responseType: "blob",
  });
  return response.data as Blob;
}

export async function importYaml(file: File): Promise<EngineConfigRequest> {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await client.post<EngineConfigRequest>(
    "/config/import-yaml",
    formData,
    {
      headers: { "Content-Type": "multipart/form-data" },
    }
  );
  return data;
}
