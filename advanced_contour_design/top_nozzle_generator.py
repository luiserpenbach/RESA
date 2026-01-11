"""
Rao Optimum Nozzle Contour Generator (Thrust Optimized Parabolic)
=================================================================
Implements the Rao thrust-optimized parabolic (TOP) nozzle design method.
This approach approximates the result of a full Variational Method of
Characteristics optimization using a circular entrance arc and a
cubic Bezier curve (parabola) for the bell.

References:
- Rao, G.V.R., "Exhaust Nozzle Contour for Optimum Thrust",
  Jet Propulsion, Vol. 28, No. 6, 1958
- Sutton & Biblarz, "Rocket Propulsion Elements"

Author: Rocket Nozzle Design Suite
"""

import numpy as np
import os
from dataclasses import dataclass
from typing import Tuple, List, Dict
from scipy.optimize import brentq
import json


# =============================================================================
# CONSTANTS AND GAS DYNAMICS
# =============================================================================

@dataclass
class GasProperties:
    """Thermodynamic properties of combustion gases."""
    gamma: float = 1.2  # Ratio of specific heats
    R: float = 350.0  # Specific gas constant [J/kg-K]
    T_c: float = 3500.0  # Chamber temperature [K]
    P_c: float = 7.0e6  # Chamber pressure [Pa]

    @property
    def a_star(self) -> float:
        """Sonic velocity at throat [m/s]."""
        return np.sqrt(self.gamma * self.R * self.T_c * (2 / (self.gamma + 1)))


def prandtl_meyer(M: float, gamma: float) -> float:
    """Prandtl-Meyer function: turning angle from M=1 to M."""
    if M <= 1.0:
        return 0.0

    gp1 = gamma + 1
    gm1 = gamma - 1

    term1 = np.sqrt(gp1 / gm1)
    term2 = np.arctan(np.sqrt(gm1 / gp1 * (M ** 2 - 1)))
    term3 = np.arctan(np.sqrt(M ** 2 - 1))

    return term1 * term2 - term3


def inverse_prandtl_meyer(nu: float, gamma: float, M_guess: float = 2.0) -> float:
    """Inverse Prandtl-Meyer function."""
    if nu <= 0:
        return 1.0

    # Maximum P-M angle (for M -> infinity)
    nu_max = (np.pi / 2) * (np.sqrt((gamma + 1) / (gamma - 1)) - 1)

    if nu >= nu_max:
        return 50.0

    def residual(M):
        return prandtl_meyer(M, gamma) - nu

    try:
        return brentq(residual, 1.0, 50.0)
    except ValueError:
        return M_guess


def mach_from_area_ratio(AR: float, gamma: float, supersonic: bool = True) -> float:
    """Calculate Mach number from area ratio A/A*."""
    if AR < 1.0:
        return 1.0  # Physically impossible for isentropic 1D flow

    gp1 = gamma + 1
    gm1 = gamma - 1
    power = gp1 / (2 * gm1)

    def area_ratio_eq(M):
        if M <= 0: return -AR
        term = (2 / gp1) * (1 + gm1 / 2 * M ** 2)
        return (1 / M) * term ** power - AR

    if supersonic:
        try:
            return brentq(area_ratio_eq, 1.0001, 50.0)
        except ValueError:
            return 1.0
    else:
        # Subsonic branch
        try:
            return brentq(area_ratio_eq, 0.0001, 0.9999)
        except ValueError:
            return 1.0


def isentropic_ratios(M: float, gamma: float) -> Dict[str, float]:
    """Calculate isentropic flow ratios."""
    gm1 = gamma - 1
    gp1 = gamma + 1

    factor = 1 + gm1 / 2 * M ** 2

    T_ratio = 1 / factor
    P_ratio = T_ratio ** (gamma / gm1)
    rho_ratio = T_ratio ** (1 / gm1)

    if M > 0:
        A_ratio = (1 / M) * ((2 / gp1) * factor) ** (gp1 / (2 * gm1))
    else:
        A_ratio = float('inf')

    return {
        'T_T0': T_ratio,
        'P_P0': P_ratio,
        'rho_rho0': rho_ratio,
        'A_Astar': A_ratio
    }


# =============================================================================
# RAO OPTIMUM NOZZLE CONTOUR
# =============================================================================

@dataclass
class RaoNozzleParams:
    """Parameters for Rao optimum nozzle design."""
    R_t: float  # Throat radius [m]
    expansion_ratio: float  # Exit area ratio A_e/A_t
    L_percent: float = 80  # Length as % of 15° cone
    gamma: float = 1.2  # Ratio of specific heats

    # Throat geometry
    R_throat_upstream: float = None  # Default: 1.5*R_t
    R_throat_downstream: float = None  # Default: 0.382*R_t

    # Convergent section
    contraction_ratio: float = 5.0
    theta_convergent: float = 35.0

    def __post_init__(self):
        if self.R_throat_upstream is None:
            self.R_throat_upstream = 1.5 * self.R_t
        if self.R_throat_downstream is None:
            self.R_throat_downstream = 0.382 * self.R_t


@dataclass
class RaoNozzleResult:
    x: np.ndarray  # Axial coordinates [m]
    r: np.ndarray  # Radial coordinates [m]
    M: np.ndarray  # Mach number
    theta: np.ndarray  # Flow angle [rad]
    nu: np.ndarray  # Prandtl-Meyer angle [rad]
    theta_n: float  # Initial expansion angle [rad]
    theta_e: float  # Exit angle [rad]
    L_nozzle: float  # Nozzle length [m]
    M_exit: float  # Exit Mach number
    x_N: float  # Inflection point position [m]
    y_N: float  # Inflection point radius [m]


class RaoOptimumNozzle:
    def __init__(self, params: RaoNozzleParams):
        self.params = params
        self.gas = GasProperties(gamma=params.gamma)

    def compute_exit_mach(self) -> float:
        return mach_from_area_ratio(self.params.expansion_ratio, self.params.gamma)

    def compute_reference_length(self) -> float:
        """Length of a 15° half-angle conical nozzle with same ER."""
        R_t = self.params.R_t
        R_e = R_t * np.sqrt(self.params.expansion_ratio)
        return (R_e - R_t) / np.tan(np.radians(15))

    def compute_optimal_angles(self) -> Tuple[float, float]:
        """Compute optimal initial (theta_n) and exit (theta_e) angles."""
        ER = self.params.expansion_ratio
        L_pct = self.params.L_percent
        M_e = self.compute_exit_mach()
        nu_e = prandtl_meyer(M_e, self.params.gamma)

        # Empirical correlations for Rao parabola angles
        if L_pct >= 75 and L_pct <= 85:
            theta_n_deg = 33 - 4.5 * np.log10(ER) + 0.5 * (80 - L_pct)
            theta_e_deg = 7 + 3 * (80 - L_pct) / 20 - 1.5 * np.log10(ER)
        elif L_pct < 75:
            theta_n_deg = 38 - 4 * np.log10(ER) + 0.8 * (75 - L_pct)
            theta_e_deg = 12 + 4 * (75 - L_pct) / 15 - 1.2 * np.log10(ER)
        else:  # > 85
            theta_n_deg = 28 - 5 * np.log10(ER) - 0.3 * (L_pct - 85)
            theta_e_deg = 5 - 0.5 * (L_pct - 85) / 15 - 1.8 * np.log10(ER)

        theta_n_deg = np.clip(theta_n_deg, 20, 50)
        theta_e_deg = np.clip(theta_e_deg, 2, 20)

        # Physical constraint check against P-M limit
        theta_n_max = np.degrees(nu_e) * 0.7
        theta_n_deg = min(theta_n_deg, theta_n_max)

        return np.radians(theta_n_deg), np.radians(theta_e_deg)

    def compute_wall_contour(self, n_points: int = 100) -> RaoNozzleResult:
        R_t = self.params.R_t
        R_td = self.params.R_throat_downstream
        ER = self.params.expansion_ratio
        gamma = self.params.gamma

        # Exit dimensions
        R_e = R_t * np.sqrt(ER)
        M_e = self.compute_exit_mach()

        # Optimization angles
        theta_n, theta_e = self.compute_optimal_angles()

        # Lengths
        L_ref = self.compute_reference_length()
        L_nozzle = (self.params.L_percent / 100) * L_ref

        # === Point N: Inflection point (End of Throat Arc) ===
        # Note: Throat is at (0, R_t). Arc curves towards flow.
        x_N = R_td * np.sin(theta_n)
        y_N = R_t + R_td * (1 - np.cos(theta_n))

        # === Point E: Exit ===
        x_E = L_nozzle
        y_E = R_e

        # === Part 1: Throat Arc (0 to theta_n) ===
        n_arc = max(5, int(n_points * 0.2))
        theta_arc = np.linspace(0, theta_n, n_arc)
        x_arc = R_td * np.sin(theta_arc)
        y_arc = R_t + R_td * (1 - np.cos(theta_arc))

        # Flow properties approx along arc (P-M expansion + area ratio)
        # Using Area ratio is more robust for wall M distribution than pure P-M
        A_ratio_arc = (y_arc / R_t) ** 2
        M_arc = np.array([mach_from_area_ratio(ar, gamma) for ar in A_ratio_arc])
        nu_arc = np.array([prandtl_meyer(m, gamma) for m in M_arc])

        # === Part 2: Parabolic (Bezier) Section (N to E) ===
        n_bell = n_points - n_arc
        P0 = np.array([x_N, y_N])
        P3 = np.array([x_E, y_E])

        # Control point lengths
        L_bez = x_E - x_N
        # Adjust fullness based on length
        if self.params.L_percent < 75:
            f1, f2 = 0.40, 0.30
        elif self.params.L_percent > 85:
            f1, f2 = 0.30, 0.40
        else:
            f1, f2 = 0.35, 0.35

        P1 = P0 + f1 * L_bez * np.array([np.cos(theta_n), np.sin(theta_n)])
        P2 = P3 - f2 * L_bez * np.array([np.cos(theta_e), np.sin(theta_e)])

        # Safety Check: prevent loops if bell is too short
        if P1[0] > P2[0]:
            # Scale back control points to midpoint
            mid_x = (P0[0] + P3[0]) / 2
            P1[0] = min(P1[0], mid_x)
            P2[0] = max(P2[0], mid_x)

        # Bezier Curve Generation
        t = np.linspace(0, 1, n_bell)

        def bezier(t, P0, P1, P2, P3):
            mt = 1 - t
            return (mt ** 3 * P0 + 3 * mt ** 2 * t * P1 +
                    3 * mt * t ** 2 * P2 + t ** 3 * P3)

        def bezier_derivative(t, P0, P1, P2, P3):
            mt = 1 - t
            return (3 * mt ** 2 * (P1 - P0) + 6 * mt * t * (P2 - P1) +
                    3 * t ** 2 * (P3 - P2))

        bell_points = np.array([bezier(ti, P0, P1, P2, P3) for ti in t])
        x_bell = bell_points[:, 0]
        y_bell = bell_points[:, 1]

        # Flow Angle along Bell (tangent angle)
        derivs = np.array([bezier_derivative(ti, P0, P1, P2, P3) for ti in t])
        theta_bell = np.arctan2(derivs[:, 1], derivs[:, 0])

        # Mach along bell via Area Ratio
        A_ratio_bell = (y_bell / R_t) ** 2
        M_bell = np.array([mach_from_area_ratio(ar, gamma) for ar in A_ratio_bell])
        nu_bell = np.array([prandtl_meyer(M, gamma) for M in M_bell])

        # === Combine Sections ===
        # We drop the first point of bell (N) because it duplicates last point of arc
        return RaoNozzleResult(
            x=np.concatenate([x_arc, x_bell[1:]]),
            r=np.concatenate([y_arc, y_bell[1:]]),
            M=np.concatenate([M_arc, M_bell[1:]]),
            theta=np.concatenate([theta_arc, theta_bell[1:]]),
            nu=np.concatenate([nu_arc, nu_bell[1:]]),
            theta_n=theta_n,
            theta_e=theta_e,
            L_nozzle=L_nozzle,
            M_exit=M_e,
            x_N=x_N,
            y_N=y_N
        )

    def compute_full_nozzle(self, n_points: int = 200) -> Dict:
        """Compute complete nozzle contour chamber to exit."""
        params = self.params
        R_t = params.R_t
        CR = params.contraction_ratio
        R_c = R_t * np.sqrt(CR)

        # Compute Rao divergent contour
        rao_result = self.compute_wall_contour(n_points // 2)

        # Convergent Geometry
        R_up = params.R_throat_upstream
        theta_conv = np.radians(params.theta_convergent)

        # 1. Downstream Convergent Arc (connects to throat at x=0)
        # Ends at throat (x=0, y=Rt), starts at angle theta_conv
        n_arc_down = 20
        theta_down = np.linspace(theta_conv, 0, n_arc_down)
        x_arc_down = -params.R_throat_downstream * np.sin(theta_down)  # Note: using R_td for smoothness
        y_arc_down = R_t + params.R_throat_downstream * (1 - np.cos(theta_down))

        # 2. Conical Section
        # Connects upstream arc to downstream arc
        # We need to find where the upstream arc ends.
        # Let's build backwards from throat.

        x_cone_end = x_arc_down[0]
        y_cone_end = y_arc_down[0]

        # 3. Upstream Arc (connects chamber to cone)
        # Center of curvature is at (x_c, R_c - R_up)
        # Tangent point is at angle (90 - theta_conv)

        # Solve for length of cone
        dy_cone = (R_c - R_up * (1 - np.cos(theta_conv))) - y_cone_end
        dx_cone = dy_cone / np.tan(theta_conv)  # Simplified

        # Rebuild arrays with proper sequence
        # Upstream Arc
        theta_up = np.linspace(np.pi / 2, np.pi / 2 - theta_conv, 20)
        x_arc_up_local = R_up * np.cos(theta_up)
        y_arc_up = (R_c - R_up) + R_up * np.sin(theta_up)

        # Shift Upstream arc x-coordinates
        x_cone_start = x_cone_end - abs(dx_cone)
        x_arc_up_end_local = x_arc_up_local[-1]
        x_shift = x_cone_start - x_arc_up_end_local
        x_arc_up = x_arc_up_local + x_shift

        # Chamber
        L_chamber = 1.5 * R_c
        x_chamber_start = x_arc_up[0] - L_chamber
        x_chamber = np.linspace(x_chamber_start, x_arc_up[0], 20)
        y_chamber = np.full_like(x_chamber, R_c)

        # Cone
        x_cone = np.linspace(x_cone_start, x_cone_end, 20)
        y_cone = np.linspace(y_arc_up[-1], y_cone_end, 20)

        # Concatenate full arrays (handling overlaps)
        x_full = np.concatenate([
            x_chamber[:-1], x_arc_up[:-1], x_cone[:-1], x_arc_down[:-1], rao_result.x
        ])
        y_full = np.concatenate([
            y_chamber[:-1], y_arc_up[:-1], y_cone[:-1], y_arc_down[:-1], rao_result.r
        ])

        # Find throat index (closest to x=0)
        throat_idx = np.abs(x_full).argmin()

        # Calculate Mach Profile
        M_full = np.ones_like(x_full)

        # Subsonic (Convergent)
        A_ratio_conv = (y_full[:throat_idx] / R_t) ** 2
        for i, ar in enumerate(A_ratio_conv):
            M_full[i] = mach_from_area_ratio(ar, params.gamma, supersonic=False)

        # Supersonic (Divergent) - Map from Rao result
        M_full[throat_idx:] = rao_result.M
        if len(M_full[throat_idx:]) != len(rao_result.M):
            # Interpolate if sizes slightly mismatch due to concat
            M_full[throat_idx:] = np.interp(
                x_full[throat_idx:], rao_result.x, rao_result.M
            )

        # Performance Calculations
        ratios = isentropic_ratios(rao_result.M_exit, params.gamma)
        Pe_Pc = ratios['P_P0']

        # Ideal Vacuum Thrust Coefficient
        gp1 = params.gamma + 1
        gm1 = params.gamma - 1
        term1 = 2 * params.gamma ** 2 / gm1
        term2 = (2 / gp1) ** (gp1 / gm1)
        term3 = 1 - Pe_Pc ** (gm1 / params.gamma)
        Cf_ideal_vac = np.sqrt(term1 * term2 * term3) + params.expansion_ratio * Pe_Pc

        # Geometric Divergence Efficiency (approx)
        # Lambda = 0.5 * (1 + cos(theta_exit)) for conical
        # For Rao, it's typically higher. Using Variational result approx:
        lambda_div = 0.5 * (1 + np.cos(rao_result.theta_e))

        Cf_actual_vac = Cf_ideal_vac * lambda_div
        Isp_vac = Cf_actual_vac * self.gas.a_star / 9.80665

        return {
            'contour': {'x': x_full, 'y': y_full, 'M': M_full},
            'geometry': {
                'R_t': R_t, 'R_e': rao_result.r[-1],
                'L_total': rao_result.x[-1] - x_chamber[0]
            },
            'performance': {
                'M_exit': rao_result.M_exit,
                'Cf_vac_ideal': Cf_ideal_vac,
                'Cf_vac_est': Cf_actual_vac,
                'Isp_vac': Isp_vac,
                'divergence_efficiency': lambda_div
            },
            'angles': {
                'theta_n': np.degrees(rao_result.theta_n),
                'theta_e': np.degrees(rao_result.theta_e)
            }
        }


def compare_contours(R_t: float, ER: float, gamma: float = 1.2,
                     L_percents: List[float] = [60, 70, 80, 90, 100]) -> Dict:
    """Compare Rao contours at different length percentages."""
    results = {}
    for L_pct in L_percents:
        params = RaoNozzleParams(R_t=R_t, expansion_ratio=ER, L_percent=L_pct, gamma=gamma)
        nozzle = RaoOptimumNozzle(params)
        res = nozzle.compute_wall_contour(100)
        results[f'{L_pct}%'] = {
            'x': res.x.tolist(), 'r': res.r.tolist(),
            'theta_n': np.degrees(res.theta_n),
            'theta_e': np.degrees(res.theta_e),
            'L': res.L_nozzle
        }
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70 + "\nRAO OPTIMUM NOZZLE GENERATOR (TOP Approximation)\n" + "=" * 70)

    # 1. Setup Parameters
    params = RaoNozzleParams(
        R_t=0.025,  # 25 mm throat radius
        expansion_ratio=15,  # Area ratio
        L_percent=80,  # 80% Bell
        gamma=1.18  # LOX/Kerosene approx
    )

    nozzle = RaoOptimumNozzle(params)
    data = nozzle.compute_full_nozzle()

    # 2. Output Results
    geo = data['geometry']
    perf = data['performance']
    ang = data['angles']

    print(f"Geometry:")
    print(f"  Throat Radius:   {geo['R_t'] * 1000:.2f} mm")
    print(f"  Exit Radius:     {geo['R_e'] * 1000:.2f} mm")
    print(f"  Total Length:    {geo['L_total'] * 1000:.2f} mm")
    print(f"\nAngles:")
    print(f"  Inflection (Tn): {ang['theta_n']:.2f} deg")
    print(f"  Exit (Te):       {ang['theta_e']:.2f} deg")
    print(f"\nPerformance (Vacuum):")
    print(f"  Exit Mach:       {perf['M_exit']:.3f}")
    print(f"  Cf (Estimated):  {perf['Cf_vac_est']:.4f}")
    print(f"  Isp (Estimated): {perf['Isp_vac']:.1f} s")
    print(f"  Div. Efficiency: {perf['divergence_efficiency'] * 100:.2f}%")

    # 3. Save Data (Using local directory)
    cwd = os.getcwd()
    file_path = os.path.join(cwd, 'rao_nozzle_data.json')

    export_data = {
        'x': data['contour']['x'].tolist(),
        'y': data['contour']['y'].tolist(),
        'M': data['contour']['M'].tolist()
    }

    with open(file_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"\nData saved to: {file_path}")