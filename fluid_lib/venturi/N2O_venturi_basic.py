import CoolProp.CoolProp as CP
import scipy.optimize as opt
import numpy as np
import math


def size_supercritical_throat(fluid, P_in_bar, T_in_K, m_dot_target, Cd=0.96):
    """
    Sizes a throat for a Supercritical Fluid (or Gas) by finding the
    Mach 1 (Sonic) condition.
    """
    P_in = P_in_bar * 1e5

    # 1. Get Inlet State (Stagnation)
    # Entropy is constant during the expansion (Isentropic assumption)
    try:
        s_in = CP.PropsSI('S', 'P', P_in, 'T', T_in_K, fluid)
        h_in = CP.PropsSI('H', 'P', P_in, 'T', T_in_K, fluid)
        rho_in = CP.PropsSI('D', 'P', P_in, 'T', T_in_K, fluid)
    except Exception as e:
        return {"error": str(e)}

    # 2. Define the Optimization Function
    # We need to find the Throat Pressure (P_throat) such that Mach Number = 1.0
    # Mach = Velocity / Speed_of_Sound

    def objective_mach_1(P_guess):
        if P_guess >= P_in or P_guess <= 0:
            return 1e6  # Penalty for invalid pressure

        # Calculate Throat State assuming Isentropic Expansion (s_throat = s_in)
        h_throat = CP.PropsSI('H', 'P', P_guess, 'S', s_in, fluid)
        a_throat = CP.PropsSI('A', 'P', P_guess, 'S', s_in, fluid)  # Speed of sound

        # Calculate Velocity from Enthalpy drop
        # h in J/kg, v = sqrt(2 * delta_h)
        delta_h = h_in - h_throat
        if delta_h < 0: return 1e6
        v_throat = math.sqrt(2 * delta_h)

        # We want (v - a) to be 0
        return abs(v_throat - a_throat)

    # 3. Solve for Critical Throat Pressure
    # Search range: between 10% and 90% of inlet pressure
    res = opt.minimize_scalar(objective_mach_1, bounds=(P_in * 0.1, P_in * 0.9), method='bounded')

    if not res.success:
        return {"error": "Could not solve for Mach 1 condition"}

    P_throat_critical = res.x

    # 4. Calculate Final Geometry at the Critical Point
    h_throat = CP.PropsSI('H', 'P', P_throat_critical, 'S', s_in, fluid)
    rho_throat = CP.PropsSI('D', 'P', P_throat_critical, 'S', s_in, fluid)
    v_throat = math.sqrt(2 * (h_in - h_throat))
    T_throat = CP.PropsSI('T', 'P', P_throat_critical, 'S', s_in, fluid)

    # Area = m_dot / (Cd * rho * v)
    area_throat = m_dot_target / (Cd * rho_throat * v_throat)
    dia_throat_mm = math.sqrt(4 * area_throat / math.pi) * 1000.0

    return {
        "fluid": fluid,
        "phase_in": CP.PhaseSI('P', P_in, 'T', T_in_K, fluid),
        "P_in_bar": P_in_bar,
        "T_in_K": T_in_K,
        "rho_in": rho_in,
        "P_throat_bar": P_throat_critical / 1e5,
        "T_throat_K": T_throat,
        "rho_throat": rho_throat,
        "v_throat": v_throat,
        "throat_dia_mm": dia_throat_mm,
        "throat_area_mm2": area_throat * 1e6,
        "Cd": Cd
    }


# --- EXECUTION ---
if __name__ == "__main__":

    # USER INPUTS
    FLUID = 'REFPROP::N2O'
    P_INLET = 75.0  # bar (Outlet of cooling jacket)
    T_INLET = 500.0  # K (Outlet of cooling jacket)
    FLOW_TARGET = 1.1  # kg/s
    CHAMBER_PRESS = 25.0  # bar

    result = size_supercritical_throat(FLUID, P_INLET, T_INLET, FLOW_TARGET)

    print("-" * 40)
    print(f"SUPERCRITICAL INJECTOR SIZING: {FLUID}")
    print("-" * 40)
    print(f"Inlet State:      {result['phase_in']}")
    print(f"Inlet Density:    {result['rho_in']:.2f} kg/m3")
    print("-" * 40)
    print(f"Throat Diameter:  {result['throat_dia_mm']:.4f} mm")
    print("-" * 40)
    print("CRITICAL FLOW CONDITIONS (Mach 1):")
    print(f"Throat Pressure:  {result['P_throat_bar']:.2f} bar")
    print(f"Throat Temp:      {result['T_throat_K']:.1f} K")
    print(f"Throat Velocity:  {result['v_throat']:.1f} m/s")
    print("-" * 40)

    # CHECK CHOKING STATUS
    if result['P_throat_bar'] > CHAMBER_PRESS:
        print(f"STATUS: CHOKED CORRECTLY")
        print(f"   Throat P ({result['P_throat_bar']:.1f} bar) > Chamber P ({CHAMBER_PRESS} bar).")
        print(f"   Flow is independent of chamber pressure.")
    else:
        print(f"STATUS: NOT CHOKED (WARNING)")
        print(f"   Throat P is lower than Chamber P.")
        print(f"   Flow will be subsonic and sensitive to chamber noise.")