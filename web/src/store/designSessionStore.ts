/**
 * Design session store for multi-module workflow coordination.
 */
import { create } from "zustand";
import type { ModuleName, ModuleStatus } from "../types/session";
import type { CoolingChannelConfig, CoolingAnalysisResponse, CoolingChannelResponse } from "../types/cooling";
import type { WallThicknessConfig, WallThicknessResponse } from "../types/structural";
import { DEFAULT_COOLING_CONFIG } from "../types/cooling";
import { DEFAULT_WALL_THICKNESS_CONFIG } from "../types/structural";

/** Downstream dependency graph */
const DOWNSTREAM: Record<string, ModuleName[]> = {
  engine: ["contour", "cooling_channels", "cooling", "wall_thickness", "performance", "feed_system"],
  contour: ["cooling_channels", "cooling", "wall_thickness"],
  cooling_channels: ["cooling", "wall_thickness"],
  cooling: ["wall_thickness"],
  wall_thickness: [],
  performance: [],
  feed_system: [],
};

interface DesignSessionState {
  sessionId: string | null;
  moduleStatus: Record<ModuleName, ModuleStatus>;

  // Module configs
  coolingConfig: CoolingChannelConfig;
  wallThicknessConfig: WallThicknessConfig;

  // Module results
  coolingChannelResult: CoolingChannelResponse | null;
  coolingAnalysisResult: CoolingAnalysisResponse | null;
  wallThicknessResult: WallThicknessResponse | null;

  // Actions
  setSessionId: (id: string | null) => void;
  setModuleStatus: (status: Record<string, string>) => void;
  markModuleCompleted: (module: ModuleName) => void;
  invalidateDownstream: (module: ModuleName) => void;

  setCoolingConfig: (config: Partial<CoolingChannelConfig>) => void;
  setCoolingChannelResult: (r: CoolingChannelResponse | null) => void;
  setCoolingAnalysisResult: (r: CoolingAnalysisResponse | null) => void;

  setWallThicknessConfig: (config: Partial<WallThicknessConfig>) => void;
  setWallThicknessResult: (r: WallThicknessResponse | null) => void;

  reset: () => void;
}

const INITIAL_STATUS: Record<ModuleName, ModuleStatus> = {
  engine: "locked",
  contour: "locked",
  cooling_channels: "locked",
  cooling: "locked",
  wall_thickness: "locked",
  performance: "locked",
  feed_system: "locked",
};

export const useDesignSessionStore = create<DesignSessionState>((set, get) => ({
  sessionId: null,
  moduleStatus: { ...INITIAL_STATUS },

  coolingConfig: { ...DEFAULT_COOLING_CONFIG },
  wallThicknessConfig: { ...DEFAULT_WALL_THICKNESS_CONFIG },

  coolingChannelResult: null,
  coolingAnalysisResult: null,
  wallThicknessResult: null,

  setSessionId: (id) => set({ sessionId: id }),

  setModuleStatus: (status) => {
    const mapped: Record<ModuleName, ModuleStatus> = { ...INITIAL_STATUS };
    for (const [k, v] of Object.entries(status)) {
      if (k in mapped) {
        mapped[k as ModuleName] = v as ModuleStatus;
      }
    }
    set({ moduleStatus: mapped });
  },

  markModuleCompleted: (module) => {
    const current = get().moduleStatus;
    set({ moduleStatus: { ...current, [module]: "completed" } });
  },

  invalidateDownstream: (module) => {
    const current = { ...get().moduleStatus };
    const toInvalidate = DOWNSTREAM[module] ?? [];
    for (const m of toInvalidate) {
      if (current[m] === "completed") {
        current[m] = "stale";
      }
    }
    set({ moduleStatus: current });
  },

  setCoolingConfig: (partial) =>
    set((s) => ({ coolingConfig: { ...s.coolingConfig, ...partial } })),
  setCoolingChannelResult: (r) => set({ coolingChannelResult: r }),
  setCoolingAnalysisResult: (r) => set({ coolingAnalysisResult: r }),

  setWallThicknessConfig: (partial) =>
    set((s) => ({ wallThicknessConfig: { ...s.wallThicknessConfig, ...partial } })),
  setWallThicknessResult: (r) => set({ wallThicknessResult: r }),

  reset: () =>
    set({
      sessionId: null,
      moduleStatus: { ...INITIAL_STATUS },
      coolingChannelResult: null,
      coolingAnalysisResult: null,
      wallThicknessResult: null,
    }),
}));
