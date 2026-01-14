"""
RESA Two-Phase N2O Regenerative Cooling Module
===============================================

A comprehensive cooling analysis module with detailed two-phase flow physics
for nitrous oxide coolant. Designed to integrate with RESA's existing architecture.

Features:
- Phase-aware heat transfer correlations (subcritical, supercritical, near-critical)
- CHF (Critical Heat Flux) calculation and safety margin tracking
- Subcooled and saturated boiling models (Chen correlation)
- Two-phase pressure drop (Lockhart-Martinelli)
- Flow regime identification and tracking
- Supercritical heat transfer deterioration warnings

Place this module in: rocket_engine/physics/cooling_n2o.py

Author: RESA Extension
References:
    - Carey, "Liquid-Vapor Phase-Change Phenomena"
    - Chen (1966), "Correlation for Boiling Heat Transfer"
    - Groeneveld et al., "CHF Lookup Table"
    - Huzel & Huang, "Modern Engineering for Design of Liquid-Propellant Rocket Engines"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable
from enum import Enum
import warnings

try:
    from CoolProp.CoolProp import PropsSI

    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False
    warnings.warn("CoolProp not available. N2O cooling analysis requires CoolProp.")


# =============================================================================
# CONSTANTS
# =============================================================================

class N2OConstants:
    """Critical properties for Nitrous Oxide"""
    T_CRIT = 309.52  # K (36.37°C)
    P_CRIT = 7.245e6  # Pa (72.45 bar)
    RHO_CRIT = 452.0  # kg/m³
    M = 44.013e-3  # kg/mol

    @classmethod
    def is_supercritical(cls, P: float) -> bool:
        return P > cls.P_CRIT

    @classmethod
    def reduced_pressure(cls, P: float) -> float:
        return P / cls.P_CRIT

    @classmethod
    def reduced_temperature(cls, T: float) -> float:
        return T / cls.T_CRIT


# =============================================================================
# ENUMS AND RESULT DATACLASSES
# =============================================================================

class FlowRegime(Enum):
    """Two-phase flow regime classification"""
    SUBCOOLED_LIQUID = "subcooled_liquid"
    SUBCOOLED_BOILING = "subcooled_boiling"
    SATURATED_BOILING = "saturated_boiling"
    ANNULAR_FLOW = "annular_flow"
    MIST_FLOW = "mist_flow"
    CHF_RISK = "chf_risk"
    POST_CHF = "post_chf"
    SUPERHEATED_VAPOR = "superheated_vapor"
    SUPERCRITICAL = "supercritical"
    PSEUDO_CRITICAL = "pseudo_critical"
    SUPERCRITICAL_GAS_LIKE = "supercritical_gas_like"


class BoilingRegime(Enum):
    """Boiling heat transfer regime"""
    SINGLE_PHASE_LIQUID = "single_phase_liquid"
    ONSET_NUCLEATE_BOILING = "onb"
    PARTIAL_NUCLEATE_BOILING = "partial_nucleate"
    FULLY_DEVELOPED_NUCLEATE = "fully_developed_nucleate"
    TRANSITION_BOILING = "transition"
    FILM_BOILING = "film_boiling"
    SINGLE_PHASE_VAPOR = "single_phase_vapor"
    SUPERCRITICAL = "supercritical"


@dataclass
class N2OFluidState:
    """Complete thermodynamic state of N2O at a point"""
    P: float  # Pressure [Pa]
    T: float  # Temperature [K]
    h: float  # Specific enthalpy [J/kg]
    rho: float  # Density [kg/m³]
    cp: float  # Specific heat [J/kg-K]
    mu: float  # Dynamic viscosity [Pa-s]
    k: float  # Thermal conductivity [W/m-K]
    phase: str  # Phase description
    quality: Optional[float] = None  # Vapor quality if two-phase

    @property
    def Pr(self) -> float:
        """Prandtl number"""
        return self.cp * self.mu / self.k if self.k > 0 else 0

    @property
    def is_supercritical(self) -> bool:
        return self.P > N2OConstants.P_CRIT

    @property
    def is_two_phase(self) -> bool:
        return self.quality is not None


@dataclass
class CoolingStationResult:
    """Results at a single station along the cooling channel"""
    # Position
    x: float  # Axial position [m]
    station_id: int  # Station index

    # Geometry at this station
    r_chamber: float  # Chamber radius at station [m]
    A_flow: float  # Coolant flow area [m²]
    D_h: float  # Hydraulic diameter [m]
    n_channels: int  # Number of channels

    # Flow conditions
    G: float  # Mass flux [kg/m²-s]
    Re: float  # Reynolds number

    # Fluid state
    fluid: N2OFluidState  # Coolant thermodynamic state

    # Heat transfer
    q_flux: float  # Heat flux from hot gas [W/m²]
    h_conv: float  # Coolant-side HTC [W/m²-K]
    T_wall_hot: float  # Hot gas side wall temp [K]
    T_wall_cold: float  # Coolant side wall temp [K]

    # Safety margins
    q_chf: float  # Critical heat flux [W/m²]
    chf_margin: float  # q_flux / q_chf (< 0.5 is safe)

    # Regime identification
    flow_regime: FlowRegime
    boiling_regime: BoilingRegime

    # Pressure
    dP_friction: float  # Friction pressure drop to this point [Pa]
    dP_acceleration: float  # Acceleration pressure drop [Pa]
    dP_total: float  # Total pressure drop [Pa]

    # Warnings
    warnings: List[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        """Check if this station is within safe operating limits"""
        return (self.chf_margin < 0.5 and
                self.T_wall_hot < 800 and  # Typical copper alloy limit
                self.flow_regime != FlowRegime.POST_CHF)


@dataclass
class CoolingAnalysisResult:
    """Complete results from cooling channel analysis"""
    # Input summary
    engine_name: str
    coolant: str
    m_dot_total: float  # Total coolant mass flow [kg/s]
    P_inlet: float  # Inlet pressure [Pa]
    T_inlet: float  # Inlet temperature [K]

    # Station-by-station results
    stations: List[CoolingStationResult]

    # Summary metrics
    T_outlet: float  # Outlet temperature [K]
    P_outlet: float  # Outlet pressure [Pa]
    dP_total: float  # Total pressure drop [Pa]
    Q_total: float  # Total heat absorbed [W]

    # Safety summary
    max_wall_temp: float  # Maximum wall temperature [K]
    min_chf_margin: float  # Minimum CHF margin (worst case)
    max_quality: float  # Maximum vapor quality reached

    # Regime summary
    regimes_encountered: List[FlowRegime]

    # Global warnings
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_feasible(self) -> bool:
        """Check if the design is feasible"""
        return (len(self.errors) == 0 and
                self.min_chf_margin < 0.5 and
                self.max_wall_temp < 800)


@dataclass
class CoolingChannelGeometry:
    """Cooling channel geometry specification"""
    # Channel dimensions (can vary with x)
    width: Callable[[float], float]  # Channel width as f(x) [m]
    height: Callable[[float], float]  # Channel height as f(x) [m]
    rib_width: Callable[[float], float]  # Rib width as f(x) [m]
    wall_thickness: float  # Chamber wall thickness [m]

    # Material properties
    k_wall: float = 350.0  # Wall thermal conductivity [W/m-K] (copper)

    def hydraulic_diameter(self, x: float) -> float:
        """Calculate hydraulic diameter at position x"""
        w = self.width(x)
        h = self.height(x)
        return 4 * w * h / (2 * (w + h))

    def flow_area(self, x: float) -> float:
        """Single channel flow area at position x"""
        return self.width(x) * self.height(x)

    def wetted_perimeter(self, x: float) -> float:
        """Single channel wetted perimeter at position x"""
        return 2 * (self.width(x) + self.height(x))


# =============================================================================
# TRANSPORT PROPERTY CORRELATIONS
# =============================================================================

def n2o_viscosity(T: float, rho: float) -> float:
    """
    N2O dynamic viscosity correlation.

    CoolProp doesn't have viscosity models for N2O, so we use
    Chapman-Enskog theory with density corrections.

    Parameters
    ----------
    T : float - Temperature [K]
    rho : float - Density [kg/m³]

    Returns
    -------
    mu : float - Dynamic viscosity [Pa-s]
    """
    # Lennard-Jones parameters for N2O
    sigma = 3.828e-10  # m
    eps_k = 232.4  # K
    M = N2OConstants.M

    T_star = T / eps_k

    # Collision integral (Neufeld correlation)
    omega = (1.16145 * T_star ** (-0.14874) +
             0.52487 * np.exp(-0.77320 * T_star) +
             2.16178 * np.exp(-2.43787 * T_star))

    # Dilute gas viscosity [Pa-s]
    mu_0 = 2.6693e-6 * np.sqrt(M * 1000 * T) / (sigma * 1e10) ** 2 / omega

    # Density correction using corresponding states
    rho_r = rho / N2OConstants.RHO_CRIT
    T_r = T / N2OConstants.T_CRIT

    if rho_r < 0.1:
        return mu_0

    # Residual viscosity (empirical)
    mu_r = mu_0 * max(0, 36.344e-6 * rho_r +
                      23.67 * rho_r ** 2 / T_r ** 0.5 -
                      7.82 * rho_r ** 3 / T_r)

    return mu_0 + mu_r


def n2o_thermal_conductivity(T: float, rho: float, cp: float) -> float:
    """
    N2O thermal conductivity correlation.

    Based on modified Eucken correlation with density correction.

    Parameters
    ----------
    T : float - Temperature [K]
    rho : float - Density [kg/m³]
    cp : float - Specific heat [J/kg-K]

    Returns
    -------
    k : float - Thermal conductivity [W/m-K]
    """
    mu = n2o_viscosity(T, rho)
    R = 8.314 / N2OConstants.M
    cv = cp - R

    # Modified Eucken for linear polyatomic
    f_int = 1.32
    k_0 = mu * (cv + f_int * R)

    # Density correction
    rho_r = rho / N2OConstants.RHO_CRIT
    T_r = T / N2OConstants.T_CRIT

    k_r = 0.05 * rho_r ** 1.5 / T_r ** 0.5 if rho_r > 0.1 else 0

    return np.clip(k_0 + k_r, 0.01, 0.5)


def n2o_surface_tension(T: float) -> float:
    """
    N2O surface tension correlation.

    Valid for subcritical temperatures only.
    """
    if T >= N2OConstants.T_CRIT:
        return 0.0

    T_r = T / N2OConstants.T_CRIT
    # Guggenheim-Katayama correlation
    sigma_0 = 0.072  # N/m at reference
    return sigma_0 * (1 - T_r) ** 1.26


# =============================================================================
# FLUID STATE CALCULATIONS
# =============================================================================

def get_n2o_state(P: float, T: float = None, h: float = None) -> N2OFluidState:
    """
    Get complete N2O thermodynamic state.

    Uses CoolProp for thermodynamic properties and custom correlations
    for transport properties.

    Parameters
    ----------
    P : float - Pressure [Pa]
    T : float, optional - Temperature [K]
    h : float, optional - Specific enthalpy [J/kg]

    Returns
    -------
    N2OFluidState with all properties
    """
    if not COOLPROP_AVAILABLE:
        raise RuntimeError("CoolProp required for N2O property calculations")

    if T is not None:
        input1, val1 = "T", T
    elif h is not None:
        input1, val1 = "H", h
    else:
        raise ValueError("Must specify either T or h")

    try:
        T_out = PropsSI("T", input1, val1, "P", P, "N2O") if h is not None else T
        h_out = PropsSI("H", input1, val1, "P", P, "N2O") if T is not None else h
        rho = PropsSI("D", input1, val1, "P", P, "N2O")
        cp = PropsSI("C", input1, val1, "P", P, "N2O")

        # Custom transport properties
        mu = n2o_viscosity(T_out, rho)
        k = n2o_thermal_conductivity(T_out, rho, cp)

        # Determine phase and quality
        quality = None
        if P < N2OConstants.P_CRIT:
            T_sat = PropsSI("T", "P", P, "Q", 0, "N2O")
            h_l = PropsSI("H", "P", P, "Q", 0, "N2O")
            h_v = PropsSI("H", "P", P, "Q", 1, "N2O")

            if h_l < h_out < h_v:
                quality = (h_out - h_l) / (h_v - h_l)
                phase = f"two_phase_x={quality:.3f}"
            elif h_out <= h_l:
                phase = "subcooled_liquid"
            else:
                phase = "superheated_vapor"
        else:
            phase = "supercritical"
            T_pc = get_pseudo_critical_temperature(P)
            if T_pc and abs(T_out - T_pc) < 5:
                phase = "near_pseudo_critical"

        return N2OFluidState(
            P=P, T=T_out, h=h_out, rho=rho, cp=cp, mu=mu, k=k,
            phase=phase, quality=quality
        )
    except Exception as e:
        raise RuntimeError(f"Property calculation failed: {e}")


def get_saturation_properties(P: float) -> Tuple[float, float, float, float, float]:
    """
    Get saturation properties at pressure P.

    Returns: (T_sat, h_l, h_v, rho_l, rho_v)
    """
    if P >= N2OConstants.P_CRIT:
        raise ValueError(f"No saturation above critical pressure ({P / 1e5:.1f} bar)")

    T_sat = PropsSI("T", "P", P, "Q", 0, "N2O")
    h_l = PropsSI("H", "P", P, "Q", 0, "N2O")
    h_v = PropsSI("H", "P", P, "Q", 1, "N2O")
    rho_l = PropsSI("D", "P", P, "Q", 0, "N2O")
    rho_v = PropsSI("D", "P", P, "Q", 1, "N2O")

    return T_sat, h_l, h_v, rho_l, rho_v


def get_pseudo_critical_temperature(P: float) -> Optional[float]:
    """Get pseudo-critical temperature for supercritical pressure."""
    if P <= N2OConstants.P_CRIT:
        return None

    # Search for cp maximum
    T_range = np.linspace(N2OConstants.T_CRIT, N2OConstants.T_CRIT + 50, 50)
    cp_max, T_pc = 0, N2OConstants.T_CRIT

    for T in T_range:
        try:
            cp = PropsSI("C", "T", T, "P", P, "N2O")
            if cp > cp_max:
                cp_max, T_pc = cp, T
        except Exception:
            # CoolProp may fail at some conditions, skip those points
            pass

    return T_pc


# =============================================================================
# HEAT TRANSFER CORRELATIONS
# =============================================================================

class HeatTransferCorrelations:
    """
    Collection of heat transfer correlations for different regimes.

    PHYSICAL NOTES:
    ===============

    SINGLE-PHASE (Gnielinski):
        - Valid for 3000 < Re < 5e6
        - More accurate than Dittus-Boelter in transition region

    SUBCOOLED BOILING (Bergles-Rohsenow + ONB):
        - ONB: Onset of Nucleate Boiling criterion
        - Partial boiling: interpolation between single-phase and full nucleate

    SATURATED BOILING (Chen):
        - Combines nucleate and convective contributions
        - S: suppression factor (nucleate suppressed at high quality)
        - F: enhancement factor (convection enhanced by two-phase)

    CRITICAL HEAT FLUX (Groeneveld/Bowring):
        - CHF is the DESIGN LIMIT for subcritical operation
        - Always maintain q" < 0.5 * q"_CHF

    SUPERCRITICAL (Jackson):
        - Property variations handled by density/Prandtl corrections
        - Watch for Heat Transfer Deterioration (HTD) near T_pc
    """

    @staticmethod
    def gnielinski(Re: float, Pr: float, k: float, D: float) -> float:
        """
        Gnielinski correlation for single-phase turbulent flow.

        Valid: 3000 < Re < 5e6, 0.5 < Pr < 2000
        """
        if Re < 2300:
            # Laminar fallback
            Nu = 3.66
        elif Re < 3000:
            # Transition interpolation
            Nu = 3.66 + (Re - 2300) / 700 * (0.023 * 3000 ** 0.8 * Pr ** 0.4 - 3.66)
        else:
            f = (0.79 * np.log(Re) - 1.64) ** (-2)
            Nu = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * (f / 8) ** 0.5 * (Pr ** (2 / 3) - 1))

        return Nu * k / D

    @staticmethod
    def dittus_boelter(Re: float, Pr: float, k: float, D: float, heating: bool = True) -> float:
        """Dittus-Boelter for turbulent single-phase (Re > 10000)."""
        n = 0.4 if heating else 0.3
        Nu = 0.023 * Re ** 0.8 * Pr ** n
        return Nu * k / D

    @staticmethod
    def onset_nucleate_boiling(P: float, G: float, D: float,
                               T_bulk: float, q_flux: float) -> Tuple[float, bool]:
        """
        Check for Onset of Nucleate Boiling (ONB).

        Returns (wall_superheat_for_ONB, is_ONB_occurring)
        """
        if P >= N2OConstants.P_CRIT:
            return 0.0, False

        T_sat = PropsSI("T", "P", P, "Q", 0, "N2O")
        rho_l = PropsSI("D", "P", P, "Q", 0, "N2O")
        h_fg = PropsSI("H", "P", P, "Q", 1, "N2O") - PropsSI("H", "P", P, "Q", 0, "N2O")
        cp_l = PropsSI("C", "P", P, "Q", 0, "N2O")
        k_l = n2o_thermal_conductivity(T_sat, rho_l, cp_l)

        sigma = n2o_surface_tension(T_sat)

        # Bergles-Rohsenow ONB criterion
        # Wall superheat required for ONB
        if sigma > 0 and h_fg > 0:
            dT_ONB = 0.556 * (q_flux / 1082) ** 0.463 * (P / 1e6) ** (-0.535)
        else:
            dT_ONB = 5.0  # Default

        # Estimate wall temperature
        mu_l = n2o_viscosity(T_sat, rho_l)
        Re_l = G * D / mu_l
        Pr_l = cp_l * mu_l / k_l
        h_sp = HeatTransferCorrelations.gnielinski(Re_l, Pr_l, k_l, D)
        T_wall_est = T_bulk + q_flux / max(h_sp, 100)

        is_ONB = (T_wall_est - T_sat) > dT_ONB

        return dT_ONB, is_ONB

    @staticmethod
    def subcooled_boiling(P: float, G: float, D: float, T_bulk: float,
                          q_flux: float) -> float:
        """
        Subcooled boiling heat transfer coefficient.

        Uses Bergles-Rohsenow partial boiling model.
        """
        T_sat = PropsSI("T", "P", P, "Q", 0, "N2O")
        rho_l = PropsSI("D", "P", P, "Q", 0, "N2O")
        cp_l = PropsSI("C", "P", P, "Q", 0, "N2O")
        mu_l = n2o_viscosity(T_sat, rho_l)
        k_l = n2o_thermal_conductivity(T_sat, rho_l, cp_l)

        Re_l = G * D / mu_l
        Pr_l = cp_l * mu_l / k_l

        # Single-phase contribution
        h_sp = HeatTransferCorrelations.gnielinski(Re_l, Pr_l, k_l, D)

        # Enhancement factor for subcooled boiling
        dT_sub = T_sat - T_bulk
        dT_wall = q_flux / h_sp

        if dT_wall > 0:
            # Partial boiling enhancement
            enhancement = 1 + 0.5 * (dT_wall / max(dT_sub, 1)) ** 2
            enhancement = min(enhancement, 5.0)  # Cap enhancement
        else:
            enhancement = 1.0

        return h_sp * enhancement

    @staticmethod
    def chen_saturated_boiling(G: float, x: float, D: float, P: float,
                               q_flux: float) -> float:
        """
        Chen correlation for saturated flow boiling.

        h_tp = S * h_nb + F * h_lo

        where S = suppression, F = enhancement
        """
        T_sat, h_l, h_v, rho_l, rho_v = get_saturation_properties(P)
        h_fg = h_v - h_l

        cp_l = PropsSI("C", "P", P, "Q", 0, "N2O")
        mu_l = n2o_viscosity(T_sat, rho_l)
        mu_v = n2o_viscosity(T_sat, rho_v)
        k_l = n2o_thermal_conductivity(T_sat, rho_l, cp_l)
        sigma = n2o_surface_tension(T_sat)

        Pr_l = cp_l * mu_l / k_l

        # Prevent division by zero
        x = np.clip(x, 1e-6, 0.999)

        # Martinelli parameter
        X_tt = ((1 - x) / x) ** 0.9 * (rho_v / rho_l) ** 0.5 * (mu_l / mu_v) ** 0.1

        # Enhancement factor F
        F = 2.35 * (1 / X_tt + 0.213) ** 0.736 if (1 / X_tt) > 0.1 else 1.0
        F = max(F, 1.0)

        # Liquid-only Reynolds
        Re_lo = G * (1 - x) * D / mu_l
        h_lo = HeatTransferCorrelations.gnielinski(Re_lo, Pr_l, k_l, D)

        # Suppression factor
        Re_tp = Re_lo * F ** 1.25
        S = 1 / (1 + 2.53e-6 * Re_tp ** 1.17)

        # Nucleate boiling (Forster-Zuber simplified)
        dT_sat = max((q_flux / 5000) ** 0.5, 1.0)

        try:
            dP = PropsSI("P", "T", T_sat + dT_sat, "Q", 0, "N2O") - P
        except Exception:
            # CoolProp may fail, use fallback pressure difference
            dP = 10000

        if sigma > 0 and h_fg > 0:
            h_nb = 0.00122 * (k_l ** 0.79 * cp_l ** 0.45 * rho_l ** 0.49 /
                              (sigma ** 0.5 * mu_l ** 0.29 * h_fg ** 0.24 * rho_v ** 0.24)) * \
                   dT_sat ** 0.24 * max(dP, 1) ** 0.75
        else:
            h_nb = 0

        return S * h_nb + F * h_lo

    @staticmethod
    def critical_heat_flux(G: float, x: float, D: float, P: float) -> float:
        """
        Critical Heat Flux calculation using Bowring correlation.

        THIS IS A CRITICAL SAFETY PARAMETER.
        Design rule: q_actual < 0.5 * q_CHF
        """
        if P >= N2OConstants.P_CRIT:
            return float('inf')  # No CHF above critical

        P_r = P / N2OConstants.P_CRIT

        h_l = PropsSI("H", "P", P, "Q", 0, "N2O")
        h_v = PropsSI("H", "P", P, "Q", 1, "N2O")
        h_fg = h_v - h_l

        # Bowring correlation (simplified)
        A = 2.317 * h_fg * G ** 0.5 / (1 + 0.0143 * D ** 0.5 * G)
        C = 0.077 * np.exp(-0.02 * P / 1e5)

        x = np.clip(x, 0, 0.99)
        q_chf = A * (1 - x) / (1 + C * G * D * max(x, 0.01) / A)

        return max(q_chf, 1e5)

    @staticmethod
    def supercritical_ht(Re: float, Pr: float, rho_bulk: float,
                         rho_wall: float, k: float, D: float) -> float:
        """
        Jackson correlation for supercritical heat transfer.

        Includes density and Prandtl corrections for property variations.
        """
        # Jackson-Hall correlation
        Nu = 0.0183 * Re ** 0.82 * Pr ** 0.5 * (rho_wall / rho_bulk) ** 0.3
        return Nu * k / D

    @staticmethod
    def check_htd_risk(T_bulk: float, T_wall: float, T_pc: float,
                       G: float, q_flux: float) -> Tuple[bool, str]:
        """
        Check for Heat Transfer Deterioration risk in supercritical flow.

        HTD occurs when:
        - Bulk is liquid-like but wall is gas-like
        - High q"/G ratio
        - Near pseudo-critical temperature
        """
        warnings = []
        htd_risk = False

        # Check if wall crosses pseudo-critical while bulk doesn't
        if T_bulk < T_pc < T_wall:
            htd_risk = True
            warnings.append("Wall crosses T_pc while bulk is liquid-like")

        # Check q"/G ratio (empirical threshold)
        q_G_ratio = q_flux / G
        if q_G_ratio > 500:  # W/kg/m
            htd_risk = True
            warnings.append(f"High q''/G ratio: {q_G_ratio:.0f} W/(kg/m)")

        # Check proximity to pseudo-critical
        if abs(T_bulk - T_pc) < 10:
            warnings.append("Operating near pseudo-critical temperature")

        return htd_risk, "; ".join(warnings) if warnings else ""


# =============================================================================
# TWO-PHASE PRESSURE DROP
# =============================================================================

class TwoPhasePressureDrop:
    """
    Two-phase pressure drop calculations using Lockhart-Martinelli.
    """

    @staticmethod
    def friction_factor(Re: float, roughness: float = 1e-6, D: float = 1e-3) -> float:
        """Darcy friction factor (Colebrook-White)"""
        if Re < 2300:
            return 64 / Re

        # Haaland approximation
        eps_D = roughness / D
        f = (-1.8 * np.log10((eps_D / 3.7) ** 1.11 + 6.9 / Re)) ** (-2)
        return f

    @staticmethod
    def single_phase_dp(G: float, rho: float, mu: float, D: float, L: float,
                        roughness: float = 1e-6) -> float:
        """Single-phase friction pressure drop"""
        Re = G * D / mu
        f = TwoPhasePressureDrop.friction_factor(Re, roughness, D)
        return f * L * G ** 2 / (2 * rho * D)

    @staticmethod
    def two_phase_multiplier(x: float, rho_l: float, rho_v: float,
                             mu_l: float, mu_v: float) -> float:
        """
        Lockhart-Martinelli two-phase multiplier (phi_lo^2).
        """
        if x <= 0:
            return 1.0
        if x >= 1:
            return (rho_l / rho_v) * (mu_v / mu_l) ** 0.2

        # Martinelli parameter
        X_tt = ((1 - x) / x) ** 0.9 * (rho_v / rho_l) ** 0.5 * (mu_l / mu_v) ** 0.1

        # Chisholm correlation for phi_lo^2
        C = 20  # Turbulent-turbulent
        phi_lo_sq = 1 + C / X_tt + 1 / X_tt ** 2

        return phi_lo_sq

    @staticmethod
    def acceleration_dp(G: float, x_in: float, x_out: float,
                        rho_l: float, rho_v: float) -> float:
        """
        Acceleration pressure drop due to vapor generation.

        This can dominate in high-quality boiling.
        """

        def void_fraction(x):
            """Zivi void fraction correlation"""
            if x <= 0:
                return 0
            if x >= 1:
                return 1
            S = (rho_l / rho_v) ** (1 / 3)  # Slip ratio
            return 1 / (1 + (1 - x) / x * rho_v / rho_l * S)

        def rho_tp(x):
            """Two-phase mixture density"""
            alpha = void_fraction(x)
            return rho_l * (1 - alpha) + rho_v * alpha

        # Momentum flux change
        rho_in = rho_tp(x_in)
        rho_out = rho_tp(x_out)

        dP_acc = G ** 2 * (1 / rho_out - 1 / rho_in)
        return max(dP_acc, 0)


# =============================================================================
# MAIN COOLING SOLVER
# =============================================================================

class N2OCoolingSolver:
    """
    1D marching solver for N2O regenerative cooling analysis.

    Features:
    - Phase-aware heat transfer correlations
    - CHF tracking and safety margin calculation
    - Two-phase pressure drop
    - Flow regime identification
    - Supercritical heat transfer deterioration warnings

    Usage:
        solver = N2OCoolingSolver(
            contour=contour_func,
            channel_geom=channel_geometry,
            q_flux_profile=heat_flux_func,
            m_dot=0.5,
            P_inlet=50e5,
            T_inlet=280
        )
        result = solver.solve()
    """

    def __init__(
            self,
            contour: Callable[[float], float],  # r_chamber(x)
            channel_geom: CoolingChannelGeometry,
            q_flux_profile: Callable[[float], float],  # q"(x) [W/m²]
            m_dot: float,  # Total mass flow [kg/s]
            P_inlet: float,  # Inlet pressure [Pa]
            T_inlet: float,  # Inlet temperature [K]
            x_start: float = 0.0,  # Start position [m]
            x_end: float = 0.1,  # End position [m]
            n_stations: int = 100,  # Number of stations
            n_channels: int = 40,  # Number of cooling channels
            flow_direction: str = "counter",  # "counter" or "co"
            engine_name: str = "Engine"
    ):
        self.contour = contour
        self.channel_geom = channel_geom
        self.q_flux_profile = q_flux_profile
        self.m_dot = m_dot
        self.P_inlet = P_inlet
        self.T_inlet = T_inlet
        self.x_start = x_start
        self.x_end = x_end
        self.n_stations = n_stations
        self.n_channels = n_channels
        self.flow_direction = flow_direction
        self.engine_name = engine_name

        # Per-channel mass flow
        self.m_dot_channel = m_dot / n_channels

    def solve(self) -> CoolingAnalysisResult:
        """
        Perform 1D marching solution along cooling channel.

        Returns complete CoolingAnalysisResult with all stations.
        """
        # Setup grid
        if self.flow_direction == "counter":
            x_grid = np.linspace(self.x_end, self.x_start, self.n_stations)
        else:
            x_grid = np.linspace(self.x_start, self.x_end, self.n_stations)

        dx = abs(x_grid[1] - x_grid[0])

        # Initialize state
        P = self.P_inlet
        state = get_n2o_state(P, T=self.T_inlet)
        h = state.h

        stations = []
        warnings = []
        errors = []
        regimes_seen = set()

        dP_friction_total = 0.0
        dP_acc_total = 0.0
        Q_total = 0.0

        for i, x in enumerate(x_grid):
            # Get geometry at this station
            r_chamber = self.contour(x)
            circumference = 2 * np.pi * r_chamber

            A_flow = self.channel_geom.flow_area(x)
            D_h = self.channel_geom.hydraulic_diameter(x)
            P_wetted = self.channel_geom.wetted_perimeter(x)

            # Mass flux
            G = self.m_dot_channel / A_flow

            # Get heat flux
            q_flux = self.q_flux_profile(x)

            # Get current fluid state
            try:
                state = get_n2o_state(P, h=h)
            except Exception as e:
                errors.append(f"Property calculation failed at x={x:.4f}m: {e}")
                break

            # Reynolds number
            Re = G * D_h / state.mu

            # Determine regime and calculate heat transfer
            flow_regime, boiling_regime, h_conv, q_chf, station_warnings = \
                self._calculate_station_heat_transfer(state, G, D_h, q_flux, P)

            warnings.extend([f"x={x:.4f}m: {w}" for w in station_warnings])
            regimes_seen.add(flow_regime)

            # Calculate wall temperatures
            T_wall_cold = state.T + q_flux / max(h_conv, 100)
            T_wall_hot = T_wall_cold + q_flux * self.channel_geom.wall_thickness / \
                         self.channel_geom.k_wall

            # CHF margin
            chf_margin = q_flux / q_chf if q_chf > 0 else 0

            # Create station result
            station = CoolingStationResult(
                x=x,
                station_id=i,
                r_chamber=r_chamber,
                A_flow=A_flow,
                D_h=D_h,
                n_channels=self.n_channels,
                G=G,
                Re=Re,
                fluid=state,
                q_flux=q_flux,
                h_conv=h_conv,
                T_wall_hot=T_wall_hot,
                T_wall_cold=T_wall_cold,
                q_chf=q_chf,
                chf_margin=chf_margin,
                flow_regime=flow_regime,
                boiling_regime=boiling_regime,
                dP_friction=dP_friction_total,
                dP_acceleration=dP_acc_total,
                dP_total=dP_friction_total + dP_acc_total,
                warnings=station_warnings
            )
            stations.append(station)

            # Check safety
            if chf_margin > 0.5:
                warnings.append(f"x={x:.4f}m: CHF margin exceeded ({chf_margin:.2f})")
            if T_wall_hot > 800:
                warnings.append(f"x={x:.4f}m: Wall temp {T_wall_hot:.0f}K exceeds limit")

            # Update for next step
            if i < self.n_stations - 1:
                # Heat absorbed
                q_absorbed = q_flux * P_wetted * dx
                Q_total += q_absorbed

                # Enthalpy rise
                dh = q_absorbed / self.m_dot_channel
                h_new = h + dh

                # Pressure drop
                quality_old = state.quality or 0

                # Get new state for pressure drop calc
                try:
                    state_new = get_n2o_state(P, h=h_new)
                    quality_new = state_new.quality or 0
                except Exception:
                    # If state calculation fails, keep previous quality
                    quality_new = quality_old

                # Friction pressure drop
                if state.quality is not None:
                    # Two-phase
                    T_sat, _, _, rho_l, rho_v = get_saturation_properties(P)
                    mu_l = n2o_viscosity(T_sat, rho_l)
                    mu_v = n2o_viscosity(T_sat, rho_v)

                    phi_sq = TwoPhasePressureDrop.two_phase_multiplier(
                        state.quality, rho_l, rho_v, mu_l, mu_v)

                    dP_f = TwoPhasePressureDrop.single_phase_dp(
                        G, rho_l, mu_l, D_h, dx) * phi_sq

                    dP_acc = TwoPhasePressureDrop.acceleration_dp(
                        G, quality_old, quality_new, rho_l, rho_v)
                else:
                    # Single phase
                    dP_f = TwoPhasePressureDrop.single_phase_dp(
                        G, state.rho, state.mu, D_h, dx)
                    dP_acc = 0

                dP_friction_total += dP_f
                dP_acc_total += dP_acc

                P -= (dP_f + dP_acc)
                h = h_new

                # Check for pressure dropping too low
                if P < 5e5:
                    errors.append(f"Pressure dropped below 5 bar at x={x:.4f}m")
                    break

        # Final state
        final_state = stations[-1].fluid if stations else None

        # Build result
        result = CoolingAnalysisResult(
            engine_name=self.engine_name,
            coolant="N2O",
            m_dot_total=self.m_dot,
            P_inlet=self.P_inlet,
            T_inlet=self.T_inlet,
            stations=stations,
            T_outlet=final_state.T if final_state else self.T_inlet,
            P_outlet=P,
            dP_total=dP_friction_total + dP_acc_total,
            Q_total=Q_total,
            max_wall_temp=max(s.T_wall_hot for s in stations) if stations else 0,
            min_chf_margin=max(s.chf_margin for s in stations) if stations else 0,
            max_quality=max(s.fluid.quality or 0 for s in stations) if stations else 0,
            regimes_encountered=list(regimes_seen),
            warnings=warnings,
            errors=errors
        )

        return result

    def _calculate_station_heat_transfer(
            self,
            state: N2OFluidState,
            G: float,
            D: float,
            q_flux: float,
            P: float
    ) -> Tuple[FlowRegime, BoilingRegime, float, float, List[str]]:
        """
        Calculate heat transfer coefficient and identify regime at a station.

        Returns: (flow_regime, boiling_regime, h_conv, q_chf, warnings)
        """
        warnings = []
        Re = G * D / state.mu

        # SUPERCRITICAL
        if state.is_supercritical:
            T_pc = get_pseudo_critical_temperature(P)

            if T_pc and abs(state.T - T_pc) < 5:
                flow_regime = FlowRegime.PSEUDO_CRITICAL
                warnings.append("Near pseudo-critical - property variations extreme")
            elif T_pc and state.T > T_pc:
                flow_regime = FlowRegime.SUPERCRITICAL_GAS_LIKE
            else:
                flow_regime = FlowRegime.SUPERCRITICAL

            boiling_regime = BoilingRegime.SUPERCRITICAL

            # Supercritical HTC
            h_conv = HeatTransferCorrelations.supercritical_ht(
                Re, state.Pr, state.rho, state.rho, state.k, D)

            # Check HTD risk
            T_wall_est = state.T + q_flux / max(h_conv, 100)
            if T_pc:
                htd_risk, htd_warning = HeatTransferCorrelations.check_htd_risk(
                    state.T, T_wall_est, T_pc, G, q_flux)
                if htd_risk:
                    warnings.append(f"HTD risk: {htd_warning}")
                    h_conv *= 0.5  # Conservative reduction

            return flow_regime, boiling_regime, h_conv, float('inf'), warnings

        # SUBCRITICAL
        T_sat = PropsSI("T", "P", P, "Q", 0, "N2O")
        q_chf = HeatTransferCorrelations.critical_heat_flux(
            G, state.quality or 0, D, P)

        # Check CHF
        if q_flux > q_chf:
            flow_regime = FlowRegime.POST_CHF
            boiling_regime = BoilingRegime.FILM_BOILING
            h_conv = 500  # Very poor film boiling
            warnings.append("POST-CHF CONDITION - WALL BURNOUT RISK")
            return flow_regime, boiling_regime, h_conv, q_chf, warnings

        if q_flux > 0.9 * q_chf:
            warnings.append(f"Approaching CHF ({q_flux / q_chf:.0%})")

        # TWO-PHASE
        if state.is_two_phase:
            x = state.quality

            if x > 0.8:
                flow_regime = FlowRegime.MIST_FLOW
            elif x > 0.3:
                flow_regime = FlowRegime.ANNULAR_FLOW
            else:
                flow_regime = FlowRegime.SATURATED_BOILING

            boiling_regime = BoilingRegime.FULLY_DEVELOPED_NUCLEATE

            h_conv = HeatTransferCorrelations.chen_saturated_boiling(
                G, x, D, P, q_flux)

            return flow_regime, boiling_regime, h_conv, q_chf, warnings

        # SUBCOOLED
        if state.T >= T_sat:
            # Superheated vapor
            flow_regime = FlowRegime.SUPERHEATED_VAPOR
            boiling_regime = BoilingRegime.SINGLE_PHASE_VAPOR
            h_conv = HeatTransferCorrelations.gnielinski(Re, state.Pr, state.k, D)
            return flow_regime, boiling_regime, h_conv, q_chf, warnings

        # Check for subcooled boiling
        dT_ONB, is_ONB = HeatTransferCorrelations.onset_nucleate_boiling(
            P, G, D, state.T, q_flux)

        if is_ONB:
            flow_regime = FlowRegime.SUBCOOLED_BOILING
            boiling_regime = BoilingRegime.PARTIAL_NUCLEATE_BOILING
            h_conv = HeatTransferCorrelations.subcooled_boiling(
                P, G, D, state.T, q_flux)
        else:
            flow_regime = FlowRegime.SUBCOOLED_LIQUID
            boiling_regime = BoilingRegime.SINGLE_PHASE_LIQUID
            h_conv = HeatTransferCorrelations.gnielinski(Re, state.Pr, state.k, D)

        return flow_regime, boiling_regime, h_conv, q_chf, warnings


# =============================================================================
# COOLING CHANNEL GENERATOR
# =============================================================================

class CoolingChannelGenerator:
    """
    Generate cooling channel geometry for a given chamber contour.

    Design philosophy:
    - Throat region: Narrow channels, high velocity for maximum CHF margin
    - Chamber: Wider channels for lower pressure drop
    - Nozzle: Transition geometry
    """

    def __init__(
            self,
            contour: Callable[[float], float],  # r(x)
            x_throat: float,
            r_throat: float,
            n_channels: int = 40
    ):
        self.contour = contour
        self.x_throat = x_throat
        self.r_throat = r_throat
        self.n_channels = n_channels

    def design_constant_velocity(
            self,
            G_target: float,  # Target mass flux [kg/m²-s]
            aspect_ratio: float = 2.0,  # height/width
            m_dot_channel: float = 0.01,  # kg/s per channel
            min_width: float = 0.5e-3,  # Minimum channel width [m]
            max_width: float = 3.0e-3  # Maximum channel width [m]
    ) -> CoolingChannelGeometry:
        """
        Design channels for approximately constant mass flux.

        Channel width varies to maintain target G while respecting limits.
        """
        # Area required for target G
        A_target = m_dot_channel / G_target

        def width(x: float) -> float:
            # Base width from target area
            w = np.sqrt(A_target / aspect_ratio)
            return np.clip(w, min_width, max_width)

        def height(x: float) -> float:
            return width(x) * aspect_ratio

        def rib_width(x: float) -> float:
            # Rib width based on local circumference
            r = self.contour(x)
            total_width = 2 * np.pi * r / self.n_channels
            return total_width - width(x)

        return CoolingChannelGeometry(
            width=width,
            height=height,
            rib_width=rib_width,
            wall_thickness=0.5e-3
        )

    def design_variable_ar(
            self,
            ar_throat: float = 3.0,  # High AR at throat for CHF margin
            ar_chamber: float = 1.5,  # Lower AR in chamber
            width_throat: float = 1.0e-3,
            width_chamber: float = 2.0e-3,
            transition_length: float = 0.02  # m
    ) -> CoolingChannelGeometry:
        """
        Design channels with variable aspect ratio.

        Higher AR at throat increases velocity and CHF margin.
        """

        def width(x: float) -> float:
            # Smooth transition around throat
            dist_from_throat = abs(x - self.x_throat)
            if dist_from_throat < transition_length:
                frac = dist_from_throat / transition_length
                return width_throat + frac * (width_chamber - width_throat)
            return width_chamber

        def aspect_ratio(x: float) -> float:
            dist_from_throat = abs(x - self.x_throat)
            if dist_from_throat < transition_length:
                frac = dist_from_throat / transition_length
                return ar_throat + frac * (ar_chamber - ar_throat)
            return ar_chamber

        def height(x: float) -> float:
            return width(x) * aspect_ratio(x)

        def rib_width(x: float) -> float:
            r = self.contour(x)
            total_width = 2 * np.pi * r / self.n_channels
            return max(total_width - width(x), 0.3e-3)

        return CoolingChannelGeometry(
            width=width,
            height=height,
            rib_width=rib_width,
            wall_thickness=0.5e-3
        )


# =============================================================================
# INTEGRATION WITH RESA
# =============================================================================

def create_cooling_solver_from_resa_config(
        engine_config,  # RESA EngineConfig object
        contour_points: np.ndarray,  # [[x, r], ...] from nozzle generator
        heat_flux_points: np.ndarray,  # [[x, q"], ...] from Bartz calculation
        n_channels: int = 40
) -> N2OCoolingSolver:
    """
    Factory function to create N2OCoolingSolver from RESA objects.

    Parameters
    ----------
    engine_config : EngineConfig
        RESA engine configuration with coolant inlet conditions
    contour_points : np.ndarray
        Chamber contour from nozzle generator, shape (N, 2)
    heat_flux_points : np.ndarray
        Heat flux profile from Bartz calculation, shape (M, 2)
    n_channels : int
        Number of cooling channels

    Returns
    -------
    N2OCoolingSolver ready to run
    """
    from scipy.interpolate import interp1d

    # Create interpolation functions
    contour_func = interp1d(
        contour_points[:, 0],
        contour_points[:, 1],
        kind='cubic',
        fill_value='extrapolate'
    )

    q_flux_func = interp1d(
        heat_flux_points[:, 0],
        heat_flux_points[:, 1],
        kind='linear',
        fill_value='extrapolate'
    )

    # Find throat location (minimum radius)
    x_throat_idx = np.argmin(contour_points[:, 1])
    x_throat = contour_points[x_throat_idx, 0]
    r_throat = contour_points[x_throat_idx, 1]

    # Generate channel geometry
    generator = CoolingChannelGenerator(
        contour=contour_func,
        x_throat=x_throat,
        r_throat=r_throat,
        n_channels=n_channels
    )

    # Estimate mass flux from total flow rate
    m_dot_total = engine_config.coolant_m_dot if hasattr(engine_config, 'coolant_m_dot') \
        else engine_config.m_dot_total * 0.3  # Default: 30% of total

    G_target = 3000  # kg/m²-s, good for CHF margin

    channel_geom = generator.design_variable_ar(
        ar_throat=3.0,
        ar_chamber=1.5,
        width_throat=1.0e-3,
        width_chamber=2.0e-3
    )

    # Create solver
    solver = N2OCoolingSolver(
        contour=contour_func,
        channel_geom=channel_geom,
        q_flux_profile=q_flux_func,
        m_dot=m_dot_total,
        P_inlet=engine_config.coolant_p_in_bar * 1e5,
        T_inlet=engine_config.coolant_t_in_k,
        x_start=contour_points[0, 0],
        x_end=contour_points[-1, 0],
        n_stations=100,
        n_channels=n_channels,
        flow_direction="counter",
        engine_name=engine_config.engine_name
    )

    return solver


# =============================================================================
# VISUALIZATION (requires plotly)
# =============================================================================

def plot_cooling_results(result: CoolingAnalysisResult):
    """Create comprehensive visualization of cooling analysis results."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise RuntimeError("Plotly required for visualization")

    x = [s.x * 1000 for s in result.stations]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Temperature Profile", "Pressure Drop",
            "Heat Transfer Coefficient", "CHF Margin",
            "Flow Regime", "Quality Profile"
        ),
        vertical_spacing=0.1
    )

    # Temperature
    fig.add_trace(go.Scatter(
        x=x, y=[s.fluid.T - 273.15 for s in result.stations],
        name='T_coolant', line=dict(color='blue')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=[s.T_wall_hot - 273.15 for s in result.stations],
        name='T_wall_hot', line=dict(color='red')
    ), row=1, col=1)

    # Pressure
    fig.add_trace(go.Scatter(
        x=x, y=[s.fluid.P / 1e5 for s in result.stations],
        name='Pressure', line=dict(color='purple')
    ), row=1, col=2)

    # HTC
    fig.add_trace(go.Scatter(
        x=x, y=[s.h_conv / 1000 for s in result.stations],
        name='h_conv', line=dict(color='green')
    ), row=2, col=1)

    # CHF Margin
    fig.add_trace(go.Scatter(
        x=x, y=[s.chf_margin for s in result.stations],
        name='q/q_CHF', line=dict(color='red')
    ), row=2, col=2)
    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  row=2, col=2, annotation_text="Safety Limit")

    # Flow regime
    regime_map = {r: i for i, r in enumerate(FlowRegime)}
    fig.add_trace(go.Scatter(
        x=x, y=[regime_map[s.flow_regime] for s in result.stations],
        mode='markers', name='Regime',
        marker=dict(color=[regime_map[s.flow_regime] for s in result.stations],
                    colorscale='Viridis')
    ), row=3, col=1)

    # Quality
    fig.add_trace(go.Scatter(
        x=x, y=[s.fluid.quality or 0 for s in result.stations],
        name='Quality', line=dict(color='orange')
    ), row=3, col=2)

    fig.update_layout(
        title=f"N2O Cooling Analysis: {result.engine_name}",
        height=900, width=1100
    )

    return fig


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Simple test case
    print("N2O Cooling Analysis Module")
    print("=" * 50)


    # Define simple chamber contour (throat at x=0.05m)
    def contour(x):
        x_throat = 0.05
        r_throat = 0.02
        r_chamber = 0.04
        if x < x_throat:
            return r_chamber - (r_chamber - r_throat) * (x / x_throat) ** 2
        else:
            return r_throat + (x - x_throat) * 0.3  # 15° half angle


    # Heat flux profile (peaked at throat)
    def q_flux(x):
        x_throat = 0.05
        q_max = 4e6  # 4 MW/m²
        sigma = 0.015
        return q_max * np.exp(-(x - x_throat) ** 2 / (2 * sigma ** 2)) + 0.5e6


    # Channel geometry
    generator = CoolingChannelGenerator(
        contour=contour,
        x_throat=0.05,
        r_throat=0.02,
        n_channels=40
    )

    channel_geom = generator.design_variable_ar()

    # Create and run solver
    solver = N2OCoolingSolver(
        contour=contour,
        channel_geom=channel_geom,
        q_flux_profile=q_flux,
        m_dot=0.5,
        P_inlet=50e5,
        T_inlet=280,
        x_start=0.0,
        x_end=0.2,
        n_stations=100,
        n_channels=30,
        engine_name="Test Engine"
    )

    result = solver.solve()

    # Print summary
    print(f"\nResults for {result.engine_name}")
    print(f"  Outlet temperature: {result.T_outlet - 273.15:.1f}°C")
    print(f"  Pressure drop: {result.dP_total / 1e5:.2f} bar")
    print(f"  Heat absorbed: {result.Q_total / 1000:.1f} kW")
    print(f"  Max wall temp: {result.max_wall_temp - 273.15:.0f}°C")
    print(f"  Max CHF margin: {result.min_chf_margin:.2f}")
    print(f"  Max quality: {result.max_quality:.3f}")
    print(f"  Regimes: {[r.value for r in result.regimes_encountered]}")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for w in result.warnings[:5]:
            print(f"  - {w}")

    if result.errors:
        print(f"\nERRORS:")
        for e in result.errors:
            print(f"  - {e}")