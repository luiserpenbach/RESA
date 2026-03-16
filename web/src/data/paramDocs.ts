/**
 * Inline documentation for engine configuration parameters.
 * Each entry is keyed by the EngineConfigRequest field name.
 */
export interface ParamDoc {
  description: string;
  range?: string;
  note?: string;
}

export const PARAM_DOCS: Record<string, ParamDoc> = {
  thrust_n: {
    description: "Target thrust delivered by the engine (sea-level or vacuum).",
    range: "100 N – 500 kN (typical hobby/test)",
    note: "Used to back-calculate throat area from Pc and c*: At = ṁ·c* / Pc.",
  },
  pc_bar: {
    description: "Stagnation pressure inside the combustion chamber.",
    range: "10 – 80 bar (amateur)  ·  80 – 300 bar (orbital)",
    note: "Higher Pc → higher Isp but requires stronger structure and higher pump pressure.",
  },
  mr: {
    description: "Oxidizer-to-fuel mass flow ratio (O/F).",
    range: "N2O/Ethanol ≈ 3.5 – 5.0  ·  LOX/RP-1 ≈ 2.4 – 2.8",
    note: "Slightly fuel-rich of stoichiometric improves cooling and reduces combustion temperature.",
  },
  eff_combustion: {
    description: "Combustion efficiency η_c* — ratio of actual c* to theoretical CEA c*.",
    range: "0.90 – 0.98 (well-designed injectors)",
    note: "Accounts for incomplete combustion, mixing inefficiency, and heat losses to walls.",
  },
  eff_nozzle_divergence: {
    description: "Nozzle divergence efficiency — corrects for the finite exit half-angle.",
    range: "0.96 – 0.99 (Rao bell at 80% length)",
    note: "λ = ½(1 + cos α_exit). Bell nozzles achieve higher λ than conical nozzles at same length.",
  },
  expansion_ratio: {
    description: "Ratio of nozzle exit area to throat area (Ae/At).",
    range: "Sea-level: 4 – 20  ·  Vacuum: 40 – 400",
    note: "Set to 0 to compute automatically from the exit pressure via isentropic relations.",
  },
  p_exit_bar: {
    description: "Nozzle exit static pressure. Used to compute expansion ratio when set manually.",
    range: "0.05 – 1.0 bar (altitude-optimised)",
    note: "Maximum thrust when Pe = Pa (ambient). Over-expansion (Pe < Pa) risks flow separation.",
  },
  L_star: {
    description: "Characteristic chamber length L* = Vc / At (combustion volume ÷ throat area).",
    range: "500 – 1500 mm (storable liquid propellants)",
    note: "Controls propellant residence time. Too short → poor combustion efficiency; too long → heavy engine.",
  },
  contraction_ratio: {
    description: "Ratio of combustion chamber cross-section to throat area (Ac/At).",
    range: "4 – 10 (typical)",
    note: "Larger ratio → lower chamber velocity, more uniform combustion, but heavier chamber.",
  },
  theta_convergent: {
    description: "Half-angle of the convergent (inlet) section connecting the chamber to the throat.",
    range: "25° – 45° (typical)",
    note: "Steeper angles shorten the inlet but increase boundary-layer separation risk near the throat.",
  },
  theta_exit: {
    description: "Exit half-angle for conical nozzles, or the initial expansion angle for bell nozzles.",
    range: "10° – 20° (conical)  ·  5° – 15° (bell exit)",
    note: "Lower angles reduce divergence losses but increase nozzle length and mass.",
  },
  bell_fraction: {
    description: "Bell nozzle length as a fraction of an equivalent 15° conical nozzle length.",
    range: "0.60 – 1.00  (0.80 is the standard trade-off)",
    note: "Rao's method optimises the bell profile for maximum thrust coefficient at this fraction.",
  },
  freeze_at_throat: {
    description: "Assume chemical equilibrium freezes at the throat (no recombination in the nozzle).",
    range: "ON = frozen flow  ·  OFF = shifting equilibrium",
    note: "Frozen Isp is conservative (lower). Equilibrium Isp is optimistic. Real engines lie between.",
  },
  fuel_injection_temp_k: {
    description: "Temperature of the fuel at the injector face.",
    range: "250 – 400 K (ambient liquid propellants)",
    note: "Affects CEA enthalpy inputs. Use the actual propellant delivery temperature.",
  },
  oxidizer_injection_temp_k: {
    description: "Temperature of the oxidizer at the injector face.",
    range: "Liquid N2O ≈ 273 K  ·  LOX ≈ 90 K",
    note: "Critical for cryogenic oxidisers. Affects thermodynamic property calculations via CEA.",
  },
};
