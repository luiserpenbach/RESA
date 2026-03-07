import { create } from "zustand";
import type { EngineConfigRequest, EngineDesignResponse } from "../types/engine";
import { DEFAULT_ENGINE_CONFIG } from "../types/engine";

interface EngineStore {
  activeConfig: EngineConfigRequest;
  lastDesignResult: EngineDesignResponse | null;
  isDirty: boolean;
  setConfig: (config: EngineConfigRequest) => void;
  setResult: (result: EngineDesignResponse) => void;
  markClean: () => void;
}

export const useEngineStore = create<EngineStore>((set) => ({
  activeConfig: DEFAULT_ENGINE_CONFIG,
  lastDesignResult: null,
  isDirty: false,

  setConfig: (config) =>
    set({ activeConfig: config, isDirty: true }),

  setResult: (result) =>
    set({ lastDesignResult: result, isDirty: false }),

  markClean: () => set({ isDirty: false }),
}));
