/**
 * Per-module methods and references shown in the collapsible Methods Panel.
 */
export interface MethodSection {
  title: string;
  lines: string[];
}

export interface ModuleDoc {
  title: string;
  sections: MethodSection[];
  reference?: string;
}

export const MODULE_DOCS: Record<string, ModuleDoc> = {
  "/engine": {
    title: "Engine Design",
    sections: [
      {
        title: "Combustion Thermodynamics",
        lines: [
          "NASA CEA (Chemical Equilibrium with Applications) via RocketCEA",
          "Computes: Tc, γ, M_mol, c* at chamber conditions",
          "c* = (1/Γ) · √(R·Tc/M_mol)  where  Γ = √(γ·(2/(γ+1))^((γ+1)/(γ-1)))",
        ],
      },
      {
        title: "Nozzle Geometry",
        lines: [
          "Throat area:   At = ṁ · c* / Pc",
          "Chamber vol.:  Vc = L* · At  →  chamber sizing from residence time",
          "Bell profile:  Rao optimum parabolic approximation",
          "Expansion:     isentropic area–Mach  A/A* = f(M, γ)",
        ],
      },
      {
        title: "Performance",
        lines: [
          "Thrust coeff:  CF = √(2γ²/(γ−1) · (2/(γ+1))^((γ+1)/(γ−1)) · [1−(Pe/Pc)^((γ−1)/γ)]) + (Pe−Pa)/Pc · Ae/At",
          "Specific Isp:  Isp = c* · CF / g₀",
          "Mass flow:     ṁ = F / (Isp · g₀)",
        ],
      },
    ],
    reference:
      "Sutton & Biblarz — Rocket Propulsion Elements, 9th ed.  ·  Rao (1958) ARS Journal",
  },

  "/cooling": {
    title: "Regenerative Cooling",
    sections: [
      {
        title: "Hot-Gas Heat Flux (Bartz)",
        lines: [
          "q = (0.026/Dt^0.2) · (μ^0.2·Cp/Pr^0.6) · (Pc·g₀/c*)^0.8 · (Dt/Rc)^0.1 · (At/A)^0.9 · σ",
          "Adiabatic wall temp: Taw = Tc · (1 + r·(γ−1)/2·M²) / (1 + (γ−1)/2·M²)",
          "Recovery factor (turbulent): r = Pr^(1/3)",
        ],
      },
      {
        title: "Coolant Side (Dittus-Boelter)",
        lines: [
          "Nu = 0.023 · Re^0.8 · Pr^0.4",
          "CoolProp real-fluid properties for N2O, ethanol, methane, water, …",
          "Marching solver integrates axially from nozzle exit → injector face",
        ],
      },
      {
        title: "Wall Thermal Model",
        lines: [
          "1-D steady-state:  q = (Taw − Tc,in) / (1/hg + tw/kw + 1/hc)",
          "Inner wall temp:   Twi = Taw − q/hg",
        ],
      },
    ],
    reference:
      "Bartz (1957) Jet Propulsion  ·  Huzel & Huang — Modern Engineering for Design of LREs",
  },

  "/structural": {
    title: "Wall Thickness (Structural)",
    sections: [
      {
        title: "Hoop Stress — Thin-Wall",
        lines: [
          "σ_hoop = Pc · r / t",
          "Required thickness: t = Pc · r / (σ_yield / FoS)",
          "Typical FoS: 1.5 – 2.0",
        ],
      },
      {
        title: "Thermal Stress",
        lines: [
          "σ_thermal = E · α · ΔT / (1 − ν)",
          "Von Mises combined: σ_VM = √(σ_hoop² + σ_th² − σ_hoop·σ_th)",
        ],
      },
    ],
    reference: "Young & Budynas — Roark's Formulas for Stress and Strain, 8th ed.",
  },

  "/performance": {
    title: "Performance Maps",
    sections: [
      {
        title: "Parametric Sweep",
        lines: [
          "CEA called at each (Pc, O/F) grid point for thermodynamic properties",
          "Isentropic relations applied for nozzle expansion at each point",
          "Contour plots: Isp_vac, c*, CF as functions of Pc and mixture ratio",
        ],
      },
      {
        title: "Throttle Analysis",
        lines: [
          "Fixed-geometry nozzle: expansion ratio fixed, Pc varies with throttle",
          "Ambient pressure sweep for altitude-optimised performance estimation",
        ],
      },
    ],
    reference: "Sutton & Biblarz — Rocket Propulsion Elements",
  },

  "/feed-system": {
    title: "Feed System",
    sections: [
      {
        title: "Line Pressure Drops",
        lines: [
          "Darcy-Weisbach: ΔP_friction = f · (L/D) · ρv²/2",
          "Minor losses: ΔP_local = Σ K · ρv²/2  (bends, valves, fittings)",
          "Friction factor: Colebrook-White equation (turbulent) or Hagen-Poiseuille (laminar)",
        ],
      },
      {
        title: "Injector & Tank",
        lines: [
          "Injector ΔP: ṁ = Cd · A · √(2ρ·ΔP_inj)",
          "Required tank pressure: P_tank = Pc + ΔP_inj + ΔP_lines",
        ],
      },
    ],
    reference: "Munson, Young & Okiishi — Fundamentals of Fluid Mechanics",
  },

  "/monte-carlo": {
    title: "Monte Carlo Analysis",
    sections: [
      {
        title: "Sampling Strategy",
        lines: [
          "Latin Hypercube Sampling (LHS) for uniform coverage of parameter space",
          "N samples (default 500) across user-defined ±σ uncertainty ranges",
          "Each sample runs a full engine design evaluation via the API",
        ],
      },
      {
        title: "Output Statistics",
        lines: [
          "Histogram + KDE with P5 / P50 / P95 percentiles",
          "Tornado chart: Pearson correlation of inputs vs key outputs",
          "Standard deviation σ and coefficient of variation CV reported",
        ],
      },
    ],
    reference:
      "McKay, Beckman & Conover (1979) — A Comparison of Three Methods for Selecting Values of Input Variables",
  },

  "/optimization": {
    title: "Design Optimization",
    sections: [
      {
        title: "Single-Objective",
        lines: [
          "Gradient-free methods (Nelder-Mead / COBYLA) via SciPy",
          "Objective: maximise Isp_vac subject to thrust and Pc constraints",
        ],
      },
      {
        title: "Multi-Objective (Pareto)",
        lines: [
          "NSGA-II genetic algorithm for Pareto front exploration",
          "Objectives: maximise Isp_vac, minimise engine dry mass",
          "Pareto front visualised as scatter plot with dominated solutions greyed out",
        ],
      },
    ],
    reference: "Deb — Multi-Objective Optimization Using Evolutionary Algorithms",
  },

  "/injector": {
    title: "Injector Design",
    sections: [
      {
        title: "Swirl Injector (LCSC / GCSC)",
        lines: [
          "Discharge coefficient: Cd = A_exit/A_port · √(1 − (A_port/A_swirl)²)",
          "Spray half-angle: α = arctan(v_tangential / v_axial)",
          "Pressure drop: ΔP = (ṁ / (Cd · A))² / (2ρ)",
          "Cd varies with swirl number S = r_swirl · A_exit / (A_port · r_exit)",
        ],
      },
    ],
    reference:
      "Bazarov, Yang & Puri — Liquid Rocket Engine Combustion Instability, AIAA Progress Series",
  },

  "/igniter": {
    title: "Torch Igniter",
    sections: [
      {
        title: "Sizing Methodology",
        lines: [
          "Miniature CEA combustion analysis at igniter Pc and MR",
          "Chamber sized via L* method (similar to main engine)",
          "HEM (Homogeneous Equilibrium Model) for two-phase N2O injector flow",
          "Spark energy requirement from ignition delay correlation",
        ],
      },
    ],
    reference:
      "Winey (2001) AIAA-2001-3450  ·  Dyer & Karabeyoglu (2007) AIAA-2007-5471",
  },

  "/tank": {
    title: "Tank Simulation",
    sections: [
      {
        title: "Pressurisation & Depletion",
        lines: [
          "Time-marching ODE for ullage pressure, pressurant gas temperature, and propellant mass",
          "Pressurant gas: ideal gas law with heat transfer to cold pressurant",
          "Propellant expulsion: mass flow from feed system pressure model",
          "Phase equilibrium for self-pressurising propellants (N2O vapour pressure)",
        ],
      },
    ],
    reference:
      "Huzel & Huang — Modern Engineering for Design of Liquid-Propellant Rocket Engines",
  },

  "/contour": {
    title: "Nozzle Contour",
    sections: [
      {
        title: "Rao Bell Profile",
        lines: [
          "Optimum parabolic approximation to the method-of-characteristics (MOC) solution",
          "Bell fraction determines nozzle length vs 15° conical reference",
          "Throat radius of curvature: Rt = 1.5 · r_t (upstream), 0.382 · r_t (downstream)",
          "3-D solid export: STL and DXF formats available",
        ],
      },
    ],
    reference:
      "Rao (1958) ARS Journal  ·  Conic sections approximation by Allman & Lilley (1979)",
  },
};
