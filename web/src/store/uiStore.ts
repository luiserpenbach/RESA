import { create } from "zustand";

export type WorkspaceTab = "dashboard" | "parameter_study" | "suggested_values";

interface UiStore {
  sidebarCollapsed: boolean;
  rightPanelCollapsed: boolean;
  navCollapsed: boolean;
  cmdPaletteOpen: boolean;
  methodsPanelOpen: boolean;
  workspaceTab: WorkspaceTab;
  lastRunTime: number | null;
  lastRunDuration: number | null;

  toggleSidebar: () => void;
  setSidebarCollapsed: (v: boolean) => void;
  toggleRightPanel: () => void;
  toggleNav: () => void;
  toggleCmdPalette: () => void;
  setCmdPaletteOpen: (v: boolean) => void;
  toggleMethodsPanel: () => void;
  setMethodsPanelOpen: (v: boolean) => void;
  setWorkspaceTab: (tab: WorkspaceTab) => void;
  setLastRunTime: (t: number | null) => void;
  setLastRunDuration: (ms: number | null) => void;
}

export const useUiStore = create<UiStore>((set) => ({
  sidebarCollapsed: false,
  rightPanelCollapsed: false,
  navCollapsed: false,
  cmdPaletteOpen: false,
  methodsPanelOpen: false,
  workspaceTab: "dashboard",
  lastRunTime: null,
  lastRunDuration: null,

  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
  setSidebarCollapsed: (v) => set({ sidebarCollapsed: v }),
  toggleRightPanel: () => set((s) => ({ rightPanelCollapsed: !s.rightPanelCollapsed })),
  toggleNav: () => set((s) => ({ navCollapsed: !s.navCollapsed })),
  toggleCmdPalette: () => set((s) => ({ cmdPaletteOpen: !s.cmdPaletteOpen })),
  setCmdPaletteOpen: (v) => set({ cmdPaletteOpen: v }),
  toggleMethodsPanel: () => set((s) => ({ methodsPanelOpen: !s.methodsPanelOpen })),
  setMethodsPanelOpen: (v) => set({ methodsPanelOpen: v }),
  setWorkspaceTab: (tab) => set({ workspaceTab: tab }),
  setLastRunTime: (t) => set({ lastRunTime: t }),
  setLastRunDuration: (ms) => set({ lastRunDuration: ms }),
}));
