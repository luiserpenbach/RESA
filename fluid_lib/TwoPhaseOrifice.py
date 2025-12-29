"""
Two-Phase Orifice Flow Models for Nitrous Oxide (N2O)
Using CoolProp for Thermophysical Properties

This module implements several two-phase flow models commonly used for
predicting mass flow rates through orifices in nitrous oxide feed systems:

1. HEM  - Homogeneous Equilibrium Model
2. SPI  - Single-Phase Incompressible (liquid-only baseline)
3. HNE  - Homogeneous Non-Equilibrium (with relaxation length)
4. Dyer - Dyer model (interpolation between SPI and HEM)

References:
- Dyer et al., "Modeling Feed System Flow Physics for Self-Pressurizing
  Propellants", AIAA 2007-5702
- Waxman et al., "Mass Flow Rate and Isolation Characteristics of Injectors
  for Use with Self-Pressurizing Oxidizers", AIAA 2013-3636

Requirements:
    pip install CoolProp

Author: Luis
Date: 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import brentq
from enum import Enum

import CoolProp.CoolProp as CP


# =============================================================================
# N2O PROPERTIES WRAPPER USING COOLPROP
# =============================================================================

class N2OProperties:
    """
    Nitrous oxide thermophysical properties using CoolProp.

    CoolProp provides high-accuracy equations of state for N2O based on
    the Span-Wagner formulation.
    """

    def __init__(self):
        self.fluid = "N2O"
        # Cache critical properties
        self.T_crit = CP.PropsSI("Tcrit", self.fluid)  # Critical temperature [K]
        self.P_crit = CP.PropsSI("Pcrit", self.fluid)  # Critical pressure [Pa]
        self.rho_crit = CP.PropsSI("rhocrit", self.fluid)  # Critical density [kg/m³]
        self.M = CP.PropsSI("M", self.fluid)  # Molar mass [kg/mol]

    def vapor_pressure(self, T: float) -> float:
        """Saturation vapor pressure [Pa] at temperature T [K]."""
        if T >= self.T_crit:
            return self.P_crit
        try:
            return CP.PropsSI("P", "T", T, "Q", 0, self.fluid)
        except:
            return self.P_crit

    def saturation_temperature(self, P: float) -> float:
        """Saturation temperature [K] at pressure P [Pa]."""
        if P >= self.P_crit:
            return self.T_crit
        try:
            return CP.PropsSI("T", "P", P, "Q", 0, self.fluid)
        except:
            return self.T_crit

    def liquid_density(self, T: float) -> float:
        """Saturated liquid density [kg/m³] at temperature T [K]."""
        if T >= self.T_crit:
            return self.rho_crit
        try:
            return CP.PropsSI("D", "T", T, "Q", 0, self.fluid)
        except:
            return self.rho_crit

    def vapor_density(self, T: float) -> float:
        """Saturated vapor density [kg/m³] at temperature T [K]."""
        if T >= self.T_crit:
            return self.rho_crit
        try:
            return CP.PropsSI("D", "T", T, "Q", 1, self.fluid)
        except:
            return self.rho_crit

    def liquid_enthalpy(self, T: float) -> float:
        """Saturated liquid specific enthalpy [J/kg] at temperature T [K]."""
        if T >= self.T_crit:
            return CP.PropsSI("H", "T", self.T_crit * 0.999, "Q", 0, self.fluid)
        try:
            return CP.PropsSI("H", "T", T, "Q", 0, self.fluid)
        except:
            return 0.0

    def vapor_enthalpy(self, T: float) -> float:
        """Saturated vapor specific enthalpy [J/kg] at temperature T [K]."""
        if T >= self.T_crit:
            return CP.PropsSI("H", "T", self.T_crit * 0.999, "Q", 1, self.fluid)
        try:
            return CP.PropsSI("H", "T", T, "Q", 1, self.fluid)
        except:
            return 0.0

    def latent_heat(self, T: float) -> float:
        """Latent heat of vaporization [J/kg] at temperature T [K]."""
        if T >= self.T_crit:
            return 0.0
        return self.vapor_enthalpy(T) - self.liquid_enthalpy(T)

    def liquid_entropy(self, T: float) -> float:
        """Saturated liquid specific entropy [J/(kg·K)] at temperature T [K]."""
        if T >= self.T_crit:
            return CP.PropsSI("S", "T", self.T_crit * 0.999, "Q", 0, self.fluid)
        try:
            return CP.PropsSI("S", "T", T, "Q", 0, self.fluid)
        except:
            return 0.0

    def vapor_entropy(self, T: float) -> float:
        """Saturated vapor specific entropy [J/(kg·K)] at temperature T [K]."""
        if T >= self.T_crit:
            return CP.PropsSI("S", "T", self.T_crit * 0.999, "Q", 1, self.fluid)
        try:
            return CP.PropsSI("S", "T", T, "Q", 1, self.fluid)
        except:
            return 0.0

    def liquid_specific_heat(self, T: float) -> float:
        """Saturated liquid specific heat Cp [J/(kg·K)] at temperature T [K]."""
        if T >= self.T_crit * 0.99:
            T = self.T_crit * 0.99
        try:
            return CP.PropsSI("C", "T", T, "Q", 0, self.fluid)
        except:
            return 2000.0  # Fallback value

    def liquid_viscosity(self, T: float) -> float:
        """Saturated liquid dynamic viscosity [Pa·s] at temperature T [K]."""
        if T >= self.T_crit * 0.99:
            T = self.T_crit * 0.99
        try:
            return CP.PropsSI("V", "T", T, "Q", 0, self.fluid)
        except:
            return 1e-4  # Fallback value

    def surface_tension(self, T: float) -> float:
        """Surface tension [N/m] at temperature T [K]."""
        if T >= self.T_crit:
            return 0.0
        try:
            return CP.PropsSI("I", "T", T, "Q", 0, self.fluid)
        except:
            # Fallback correlation
            T_r = T / self.T_crit
            return 0.0293 * (1 - T_r) ** 1.26

    def get_properties_at_Px(self, P: float, x: float) -> dict:
        """
        Get all relevant properties at pressure P and quality x.

        Args:
            P: Pressure [Pa]
            x: Vapor quality (0 = saturated liquid, 1 = saturated vapor)

        Returns:
            Dictionary with T, rho, h, s
        """
        try:
            T = CP.PropsSI("T", "P", P, "Q", x, self.fluid)
            rho = CP.PropsSI("D", "P", P, "Q", x, self.fluid)
            h = CP.PropsSI("H", "P", P, "Q", x, self.fluid)
            s = CP.PropsSI("S", "P", P, "Q", x, self.fluid)
            return {"T": T, "rho": rho, "h": h, "s": s}
        except:
            return None

    def get_quality_from_entropy(self, P: float, s: float) -> float:
        """
        Calculate quality x from pressure and specific entropy (isentropic).

        Args:
            P: Pressure [Pa]
            s: Specific entropy [J/(kg·K)]

        Returns:
            Quality x (0 to 1)
        """
        try:
            s_l = CP.PropsSI("S", "P", P, "Q", 0, self.fluid)
            s_v = CP.PropsSI("S", "P", P, "Q", 1, self.fluid)

            if s <= s_l:
                return 0.0
            elif s >= s_v:
                return 1.0
            else:
                return (s - s_l) / (s_v - s_l)
        except:
            return 0.0

    def get_quality_from_enthalpy(self, P: float, h: float) -> float:
        """
        Calculate quality x from pressure and specific enthalpy (isenthalpic).

        Args:
            P: Pressure [Pa]
            h: Specific enthalpy [J/kg]

        Returns:
            Quality x (0 to 1)
        """
        try:
            h_l = CP.PropsSI("H", "P", P, "Q", 0, self.fluid)
            h_v = CP.PropsSI("H", "P", P, "Q", 1, self.fluid)

            if h <= h_l:
                return 0.0
            elif h >= h_v:
                return 1.0
            else:
                return (h - h_l) / (h_v - h_l)
        except:
            return 0.0

    def two_phase_density(self, P: float, x: float) -> float:
        """
        Two-phase mixture density [kg/m³] at pressure P and quality x.
        Uses homogeneous model: 1/rho = x/rho_v + (1-x)/rho_l
        """
        try:
            rho_l = CP.PropsSI("D", "P", P, "Q", 0, self.fluid)
            rho_v = CP.PropsSI("D", "P", P, "Q", 1, self.fluid)

            if x <= 0:
                return rho_l
            elif x >= 1:
                return rho_v

            v_mix = (1 - x) / rho_l + x / rho_v
            return 1 / v_mix
        except:
            return 100.0  # Fallback


# =============================================================================
# HOMOGENEOUS EQUILIBRIUM MODEL (HEM)
# =============================================================================

class HEMModel:
    """
    Homogeneous Equilibrium Model for two-phase orifice flow.

    Assumptions:
    - Thermal equilibrium between phases
    - Mechanical equilibrium (same velocity)
    - Isentropic expansion to throat

    The HEM model gives the theoretical minimum mass flux for
    flashing flow and serves as a lower bound.
    """

    def __init__(self, props: N2OProperties = None):
        self.props = props or N2OProperties()

    def mass_flux_at_pressure(self, P_1: float, T_1: float, P_2: float) -> float:
        """
        Calculate mass flux [kg/(m²·s)] for isentropic expansion from
        upstream conditions (P_1, T_1) to downstream pressure P_2.

        Uses energy equation with isentropic process:
        G = rho_2 * sqrt(2 * (h_1 - h_2))

        where state 2 is found by following constant entropy from state 1.
        """
        if P_2 >= P_1:
            return 0.0

        # Upstream conditions (saturated or subcooled liquid)
        try:
            h_1 = CP.PropsSI("H", "P", P_1, "T", T_1, self.props.fluid)
            s_1 = CP.PropsSI("S", "P", P_1, "T", T_1, self.props.fluid)
        except:
            # If subcooled state fails, use saturated liquid
            h_1 = self.props.liquid_enthalpy(T_1)
            s_1 = self.props.liquid_entropy(T_1)

        # Check if P_2 is valid
        T_triple = 182.33  # N2O triple point
        P_min = self.props.vapor_pressure(T_triple + 5)

        if P_2 < P_min:
            P_2 = P_min

        # Downstream state at constant entropy
        try:
            # Get saturation properties at P_2
            s_l_2 = CP.PropsSI("S", "P", P_2, "Q", 0, self.props.fluid)
            s_v_2 = CP.PropsSI("S", "P", P_2, "Q", 1, self.props.fluid)

            # Determine quality from entropy
            if s_1 <= s_l_2:
                # Still subcooled liquid
                x_2 = 0.0
                h_2 = CP.PropsSI("H", "P", P_2, "Q", 0, self.props.fluid)
                rho_2 = CP.PropsSI("D", "P", P_2, "Q", 0, self.props.fluid)
            elif s_1 >= s_v_2:
                # Superheated vapor
                x_2 = 1.0
                h_2 = CP.PropsSI("H", "P", P_2, "S", s_1, self.props.fluid)
                rho_2 = CP.PropsSI("D", "P", P_2, "S", s_1, self.props.fluid)
            else:
                # Two-phase region
                x_2 = (s_1 - s_l_2) / (s_v_2 - s_l_2)
                h_l_2 = CP.PropsSI("H", "P", P_2, "Q", 0, self.props.fluid)
                h_v_2 = CP.PropsSI("H", "P", P_2, "Q", 1, self.props.fluid)
                h_2 = h_l_2 + x_2 * (h_v_2 - h_l_2)
                rho_2 = self.props.two_phase_density(P_2, x_2)
        except Exception as e:
            return 0.0

        # Energy equation
        delta_h = h_1 - h_2

        if delta_h <= 0:
            return 0.0

        G = rho_2 * np.sqrt(2 * delta_h)

        return G

    def critical_mass_flux(self, P_1: float, T_1: float,
                           n_points: int = 100) -> Tuple[float, float]:
        """
        Find critical (maximum) mass flux and corresponding throat pressure.

        Returns:
            G_crit: Critical mass flux [kg/(m²·s)]
            P_crit: Critical (throat) pressure [Pa]
        """
        # Search range
        T_triple = 182.33
        P_min = self.props.vapor_pressure(T_triple + 10)
        P_max = P_1 * 0.99

        if P_min >= P_max:
            return 0.0, P_1

        P_array = np.linspace(P_min, P_max, n_points)
        G_array = np.array([self.mass_flux_at_pressure(P_1, T_1, P)
                            for P in P_array])

        # Find maximum
        idx_max = np.argmax(G_array)
        G_crit = G_array[idx_max]
        P_crit = P_array[idx_max]

        # Refine with finer search
        if 0 < idx_max < len(P_array) - 1:
            P_low = P_array[max(0, idx_max - 2)]
            P_high = P_array[min(len(P_array) - 1, idx_max + 2)]
            P_fine = np.linspace(P_low, P_high, 30)
            G_fine = np.array([self.mass_flux_at_pressure(P_1, T_1, P)
                               for P in P_fine])
            idx_fine = np.argmax(G_fine)
            G_crit = G_fine[idx_fine]
            P_crit = P_fine[idx_fine]

        return G_crit, P_crit

    def mass_flow_rate(self, P_1: float, T_1: float, P_2: float,
                       A: float, Cd: float = 0.65) -> dict:
        """
        Calculate mass flow rate through an orifice.

        Args:
            P_1: Upstream pressure [Pa]
            T_1: Upstream temperature [K]
            P_2: Downstream pressure [Pa]
            A: Orifice area [m²]
            Cd: Discharge coefficient [-]

        Returns:
            Dictionary with flow results
        """
        # Get critical conditions
        G_crit, P_crit = self.critical_mass_flux(P_1, T_1)

        # Determine if flow is choked
        if P_2 <= P_crit:
            G = G_crit
            P_throat = P_crit
            choked = True
        else:
            G = self.mass_flux_at_pressure(P_1, T_1, P_2)
            P_throat = P_2
            choked = False

        mdot = Cd * A * G

        # Throat conditions
        try:
            T_throat = self.props.saturation_temperature(P_throat)
            h_1 = self.props.liquid_enthalpy(T_1)
            x_throat = self.props.get_quality_from_enthalpy(P_throat, h_1)
            rho_throat = self.props.two_phase_density(P_throat, x_throat)
        except:
            T_throat = T_1
            x_throat = 0.0
            rho_throat = self.props.liquid_density(T_1)

        return {
            'mdot': mdot,
            'G': G,
            'G_crit': G_crit,
            'P_crit': P_crit,
            'P_throat': P_throat,
            'T_throat': T_throat,
            'x_throat': x_throat,
            'rho_throat': rho_throat,
            'choked': choked,
            'model': 'HEM'
        }


# =============================================================================
# SINGLE-PHASE INCOMPRESSIBLE MODEL (SPI)
# =============================================================================

class SPIModel:
    """
    Single-Phase Incompressible Model.

    Assumes no phase change - liquid remains liquid throughout.
    This is the upper bound for mass flow rate.

    G_SPI = sqrt(2 * rho_l * (P_1 - P_2))
    """

    def __init__(self, props: N2OProperties = None):
        self.props = props or N2OProperties()

    def mass_flux(self, P_1: float, T_1: float, P_2: float) -> float:
        """Calculate single-phase incompressible mass flux [kg/(m²·s)]."""
        rho_l = self.props.liquid_density(T_1)
        delta_P = P_1 - P_2

        if delta_P <= 0:
            return 0.0

        return np.sqrt(2 * rho_l * delta_P)

    def mass_flow_rate(self, P_1: float, T_1: float, P_2: float,
                       A: float, Cd: float = 0.65) -> dict:
        """Calculate mass flow rate through an orifice."""
        rho_l = self.props.liquid_density(T_1)
        G = self.mass_flux(P_1, T_1, P_2)
        mdot = Cd * A * G

        v_throat = G / rho_l if G > 0 else 0.0

        return {
            'mdot': mdot,
            'G': G,
            'rho_l': rho_l,
            'v_throat': v_throat,
            'model': 'SPI'
        }


# =============================================================================
# DYER MODEL (INTERPOLATION)
# =============================================================================

class DyerModel:
    """
    Dyer Model for two-phase orifice flow.

    Interpolates between SPI and HEM based on non-equilibrium parameter k:
    G_Dyer = (1 - k) * G_SPI + k * G_HEM

    Reference: Dyer et al., AIAA 2007-5702
    """

    def __init__(self, props: N2OProperties = None):
        self.props = props or N2OProperties()
        self.hem = HEMModel(props)
        self.spi = SPIModel(props)

    def non_equilibrium_parameter(self, P_1: float, T_1: float, P_2: float) -> float:
        """
        Calculate the Dyer non-equilibrium parameter k.

        k = 0: No vaporization (SPI limit)
        k = 1: Full equilibrium (HEM limit)
        """
        P_sat = self.props.vapor_pressure(T_1)

        # Downstream above saturation - no vaporization
        if P_2 >= P_sat:
            return 0.0

        # Subcooled upstream
        if P_1 > P_sat:
            delta_P_vap = P_sat - P_2
            delta_P_total = P_1 - P_2

            if delta_P_total <= 0:
                return 0.0

            kappa = np.clip(delta_P_vap / delta_P_total, 0, 1)
            k = kappa ** 0.5
        else:
            # Saturated upstream
            k = 1.0

        return k

    def mass_flux(self, P_1: float, T_1: float, P_2: float) -> Tuple[float, dict]:
        """Calculate Dyer model mass flux."""
        G_spi = self.spi.mass_flux(P_1, T_1, P_2)
        G_hem_crit, P_crit = self.hem.critical_mass_flux(P_1, T_1)

        if P_2 <= P_crit:
            G_hem = G_hem_crit
            choked = True
        else:
            G_hem = self.hem.mass_flux_at_pressure(P_1, T_1, P_2)
            choked = False

        k = self.non_equilibrium_parameter(P_1, T_1, P_2)
        G = (1 - k) * G_spi + k * G_hem

        info = {
            'G_SPI': G_spi,
            'G_HEM': G_hem,
            'k': k,
            'choked_HEM': choked,
            'P_crit_HEM': P_crit
        }

        return G, info

    def mass_flow_rate(self, P_1: float, T_1: float, P_2: float,
                       A: float, Cd: float = 0.65) -> dict:
        """Calculate mass flow rate using Dyer model."""
        G, info = self.mass_flux(P_1, T_1, P_2)
        mdot = Cd * A * G

        return {'mdot': mdot, 'G': G, **info, 'model': 'Dyer'}


# =============================================================================
# HOMOGENEOUS NON-EQUILIBRIUM MODEL (HNE)
# =============================================================================

class HNEModel:
    """
    Homogeneous Non-Equilibrium Model.

    Accounts for finite rate of phase change using relaxation length.

    Reference: Henry & Fauske, J. Heat Transfer, 1971
    """

    def __init__(self, props: N2OProperties = None, L_orifice: float = 0.001):
        self.props = props or N2OProperties()
        self.hem = HEMModel(props)
        self.L_orifice = L_orifice

    def relaxation_length(self, T: float, P: float, G: float) -> float:
        """Estimate relaxation length for bubble growth [m]."""
        if G <= 0:
            return 0.001

        rho_l = self.props.liquid_density(T)
        sigma = self.props.surface_tension(T)

        v = G / rho_l
        We = rho_l * v ** 2 * 1e-4 / max(sigma, 1e-6)
        L_relax = 0.5e-3 * (1 + 0.1 * We ** 0.5)

        return max(L_relax, 0.1e-3)

    def non_equilibrium_factor(self, T: float, P: float, G: float) -> float:
        """Calculate non-equilibrium factor N = 1 - exp(-L/L_relax)."""
        L_relax = self.relaxation_length(T, P, G)
        N = 1 - np.exp(-self.L_orifice / L_relax)
        return np.clip(N, 0, 1)

    def mass_flux(self, P_1: float, T_1: float, P_2: float) -> Tuple[float, dict]:
        """Calculate HNE model mass flux."""
        G_hem_crit, P_crit = self.hem.critical_mass_flux(P_1, T_1)
        spi = SPIModel(self.props)
        G_spi = spi.mass_flux(P_1, T_1, P_2)

        if P_2 <= P_crit:
            G_hem = G_hem_crit
            P_throat = P_crit
        else:
            G_hem = self.hem.mass_flux_at_pressure(P_1, T_1, P_2)
            P_throat = P_2

        # Iterate for self-consistent solution
        G_old = G_spi
        for _ in range(10):
            N = self.non_equilibrium_factor(T_1, P_throat, G_old)
            G_new = G_spi - N * (G_spi - G_hem)
            if abs(G_new - G_old) / max(G_new, 1) < 1e-4:
                break
            G_old = G_new

        G = G_new
        info = {
            'G_SPI': G_spi,
            'G_HEM': G_hem,
            'N': N,
            'L_relax': self.relaxation_length(T_1, P_throat, G),
            'L_orifice': self.L_orifice
        }

        return G, info

    def mass_flow_rate(self, P_1: float, T_1: float, P_2: float,
                       A: float, Cd: float = 0.65) -> dict:
        """Calculate mass flow rate using HNE model."""
        G, info = self.mass_flux(P_1, T_1, P_2)
        mdot = Cd * A * G

        return {'mdot': mdot, 'G': G, **info, 'model': 'HNE'}


# =============================================================================
# COMPOSITE MODEL SELECTOR
# =============================================================================

class TwoPhaseOrificeFlow:
    """Unified interface for two-phase orifice flow calculations."""

    class Model(Enum):
        HEM = "HEM"
        SPI = "SPI"
        DYER = "Dyer"
        HNE = "HNE"

    def __init__(self, props: N2OProperties = None):
        self.props = props or N2OProperties()
        self.hem = HEMModel(self.props)
        self.spi = SPIModel(self.props)
        self.dyer = DyerModel(self.props)
        self.hne = HNEModel(self.props)

    def calculate(self, model: Model, P_1: float, T_1: float, P_2: float,
                  A: float, Cd: float = 0.65, **kwargs) -> dict:
        """Calculate mass flow rate using specified model."""
        if model == self.Model.HEM:
            return self.hem.mass_flow_rate(P_1, T_1, P_2, A, Cd)
        elif model == self.Model.SPI:
            return self.spi.mass_flow_rate(P_1, T_1, P_2, A, Cd)
        elif model == self.Model.DYER:
            return self.dyer.mass_flow_rate(P_1, T_1, P_2, A, Cd)
        elif model == self.Model.HNE:
            if 'L_orifice' in kwargs:
                self.hne.L_orifice = kwargs['L_orifice']
            return self.hne.mass_flow_rate(P_1, T_1, P_2, A, Cd)
        else:
            raise ValueError(f"Unknown model: {model}")

    def compare_all(self, P_1: float, T_1: float, P_2: float,
                    A: float, Cd: float = 0.65, L_orifice: float = 0.001) -> dict:
        """Compare results from all models."""
        self.hne.L_orifice = L_orifice

        results = {
            'HEM': self.hem.mass_flow_rate(P_1, T_1, P_2, A, Cd),
            'SPI': self.spi.mass_flow_rate(P_1, T_1, P_2, A, Cd),
            'Dyer': self.dyer.mass_flow_rate(P_1, T_1, P_2, A, Cd),
            'HNE': self.hne.mass_flow_rate(P_1, T_1, P_2, A, Cd)
        }

        results['summary'] = {
            'mdot_HEM': results['HEM']['mdot'],
            'mdot_SPI': results['SPI']['mdot'],
            'mdot_Dyer': results['Dyer']['mdot'],
            'mdot_HNE': results['HNE']['mdot'],
            'P_1': P_1,
            'T_1': T_1,
            'P_2': P_2,
            'P_sat': self.props.vapor_pressure(T_1)
        }

        return results

    def sweep_pressure_ratio(self, P_1: float, T_1: float, A: float,
                             Cd: float = 0.65, n_points: int = 50) -> dict:
        """Sweep downstream pressure and compute mass flow for all models."""
        P_min = self.props.vapor_pressure(190)
        P_2_array = np.linspace(P_min, P_1 * 0.99, n_points)

        results = {
            'P_2': P_2_array,
            'PR': P_2_array / P_1,
            'mdot_HEM': np.zeros(n_points),
            'mdot_SPI': np.zeros(n_points),
            'mdot_Dyer': np.zeros(n_points),
            'mdot_HNE': np.zeros(n_points),
        }

        for i, P_2 in enumerate(P_2_array):
            results['mdot_HEM'][i] = self.hem.mass_flow_rate(P_1, T_1, P_2, A, Cd)['mdot']
            results['mdot_SPI'][i] = self.spi.mass_flow_rate(P_1, T_1, P_2, A, Cd)['mdot']
            results['mdot_Dyer'][i] = self.dyer.mass_flow_rate(P_1, T_1, P_2, A, Cd)['mdot']
            results['mdot_HNE'][i] = self.hne.mass_flow_rate(P_1, T_1, P_2, A, Cd)['mdot']

        return results


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_calculation():
    """Example calculations for N2O injector orifice."""
    print("=" * 70)
    print("TWO-PHASE ORIFICE FLOW MODELS FOR N2O (Using CoolProp)")
    print("=" * 70)

    # Initialize
    props = N2OProperties()
    flow = TwoPhaseOrificeFlow(props)

    print(f"\nCoolProp N2O Critical Properties:")
    print(f"  T_crit = {props.T_crit:.2f} K ({props.T_crit - 273.15:.2f} °C)")
    print(f"  P_crit = {props.P_crit / 1e5:.2f} bar")
    print(f"  rho_crit = {props.rho_crit:.1f} kg/m³")

    # Orifice parameters
    d_orifice = 1.5e-3
    A_orifice = np.pi * (d_orifice / 2) ** 2
    Cd = 0.65

    # =========================================================================
    # CASE 1: SATURATED UPSTREAM
    # =========================================================================
    print(f"\n{'CASE 1: SATURATED LIQUID UPSTREAM':=^70}")

    T_tank = 293.15
    P_sat = props.vapor_pressure(T_tank)
    P_tank = P_sat
    P_chamber = 20e5

    print(f"\n  Tank temperature:   {T_tank - 273.15:.1f} °C")
    print(f"  Tank pressure:      {P_tank / 1e5:.2f} bar (saturated)")
    print(f"  Chamber pressure:   {P_chamber / 1e5:.1f} bar")
    print(f"  Liquid density:     {props.liquid_density(T_tank):.1f} kg/m³")
    print(f"  Latent heat:        {props.latent_heat(T_tank) / 1e3:.1f} kJ/kg")

    results = flow.compare_all(P_tank, T_tank, P_chamber, A_orifice, Cd)

    print(f"\n  {'Model':<12} {'Mass Flow [g/s]':>16} {'Mass Flux [kg/m²s]':>20}")
    print(f"  {'-' * 12} {'-' * 16} {'-' * 20}")
    for model in ['SPI', 'Dyer', 'HNE', 'HEM']:
        mdot = results[model]['mdot']
        G = results[model]['G']
        print(f"  {model:<12} {mdot * 1000:>16.2f} {G:>20.0f}")

    print(f"\n  HEM Details:")
    print(f"    Critical pressure:  {results['HEM']['P_crit'] / 1e5:.2f} bar")
    print(f"    Throat quality:     {results['HEM']['x_throat'] * 100:.1f}%")
    print(f"    Flow choked:        {'Yes' if results['HEM']['choked'] else 'No'}")

    # =========================================================================
    # CASE 2: SUBCOOLED UPSTREAM
    # =========================================================================
    print(f"\n{'CASE 2: SUBCOOLED LIQUID UPSTREAM':=^70}")

    T_tank = 293.15
    P_tank = 60e5
    P_chamber = 20e5
    P_sat = props.vapor_pressure(T_tank)

    print(f"\n  Tank temperature:   {T_tank - 273.15:.1f} °C")
    print(f"  Tank pressure:      {P_tank / 1e5:.1f} bar")
    print(f"  Saturation pressure:{P_sat / 1e5:.2f} bar")
    print(f"  Subcooling:         {(P_tank - P_sat) / 1e5:.2f} bar")
    print(f"  Chamber pressure:   {P_chamber / 1e5:.1f} bar")

    results = flow.compare_all(P_tank, T_tank, P_chamber, A_orifice, Cd)

    print(f"\n  {'Model':<12} {'Mass Flow [g/s]':>16} {'Mass Flux [kg/m²s]':>20}")
    print(f"  {'-' * 12} {'-' * 16} {'-' * 20}")
    for model in ['SPI', 'Dyer', 'HNE', 'HEM']:
        mdot = results[model]['mdot']
        G = results[model]['G']
        print(f"  {model:<12} {mdot * 1000:>16.2f} {G:>20.0f}")

    print(f"\n  Dyer Model: k = {results['Dyer']['k']:.3f}")
    print(f"    ({100 * (1 - results['Dyer']['k']):.0f}% SPI + {100 * results['Dyer']['k']:.0f}% HEM)")

    return results


if __name__ == "__main__":
    example_calculation()
    print("\n" + "=" * 70)
    print("CALCULATIONS COMPLETE")
    print("=" * 70)