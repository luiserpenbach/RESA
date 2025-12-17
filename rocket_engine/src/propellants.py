import CoolProp.CoolProp as CP
from dataclasses import dataclass
from typing import Optional, Dict, Union
import numpy as np


@dataclass
class FluidState:
    """
    snapshot of the fluid's thermodynamic and transport state.
    Used to pass complete fluid data between solvers without re-querying CoolProp.
    """
    pressure: float  # [Pa]
    temperature: float  # [K]
    enthalpy: float  # [J/kg]
    density: float  # [kg/m^3]
    specific_heat: float  # [J/(kg*K)] (Cp)
    viscosity: float  # [Pa*s]
    conductivity: float  # [W/(m*K)]
    prandtl: float  # [-]
    quality: float  # [-] (-1 if subcooled/supercritical, 0-1 if two-phase)

    @property
    def kinematic_viscosity(self) -> float:
        return self.viscosity / self.density


class Propellant:
    """
    Wrapper for CoolProp fluid properties.
    Handles aliases (e.g., 'Ethanol90'), mixture definitions, and robust property lookups.
    """

    # Registry of common rocket propellant aliases to CoolProp strings
    # Note: Ethanol mixtures are approximated by mole fraction in CoolProp strings usually,
    # but here we preserve the user's explicit ratio logic or define standard mixtures.
    # For high fidelity, use REFPROP backend if available.
    ALIASES = {
        "Ethanol": "Ethanol",
        "Ethanol90": "Ethanol[0.866]&Water[0.134]",  # Approx mole fraction for 90% wt ethanol
        "Ethanol80": "Ethanol[0.735]&Water[0.265]",  # Approx mole fraction for 80% wt ethanol
        "Ethanol75": "Ethanol[0.67]&Water[0.33]",
        "N2O": "NitrousOxide",
        "Nitrous": "NitrousOxide",
        "LOX": "Oxygen",
        "Oxygen": "Oxygen",
        "MMH": "n-Hydrazine",  # Approx? MMH data is scarce in std CoolProp
        "RP1": "n-Dodecane",  # Common surrogate. Real RP-1 requires specific libraries.
        "Water": "Water",
        "IPA": "Isopropanol"
    }

    def __init__(self, name: str, backend: str = "REFPROP"):
        """
        Args:
            name: Fluid name (e.g. "Ethanol90", "N2O")
            backend: CoolProp backend ("HEOS", "REFPROP", "BICUBIC", etc.)
        """
        self.original_name = name
        self.backend = backend
        self.fluid_string = self._resolve_name(name)

        # Validate fluid loads correctly
        try:
            # Quick check at standard conditions
            CP.PropsSI('D', 'T', 298.15, 'P', 101325, self.fluid_string)
        except ValueError as e:
            print(f"Warning: Could not initialize fluid '{self.fluid_string}'. Error: {e}")
            # We don't raise immediately to allow for custom REFPROP strings that might fail at STP

    def _resolve_name(self, name: str) -> str:
        """Resolves aliases to CoolProp-compatible strings."""
        if name in self.ALIASES:
            return self.ALIASES[name]

        # Handle "REFPROP::" prefix explicitly if user passed it
        if name.startswith("REFPROP::") or "&" in name:
            return name

        return name

    def get_state(self, pressure: float, enthalpy: float = None, temperature: float = None) -> FluidState:
        """
        Retrieves full transport properties state.
        Must provide either Enthalpy (preferred for phase change) or Temperature.

        Args:
            pressure: [Pa]
            enthalpy: [J/kg] (Optional)
            temperature: [K] (Optional)

        Returns:
            FluidState object
        """
        if enthalpy is not None:
            return self._get_state_from_ph(pressure, enthalpy)
        elif temperature is not None:
            return self._get_state_from_pt(pressure, temperature)
        else:
            raise ValueError("Must provide either enthalpy or temperature.")

    def _get_state_from_ph(self, p: float, h: float) -> FluidState:
        try:
            # Batch query could be optimized, but individual calls are safer for error handling
            # Note: Using 'D', 'V', 'L' etc.

            # 1. Basic State
            T = CP.PropsSI('T', 'P', p, 'H', h, self.fluid_string)
            rho = CP.PropsSI('D', 'P', p, 'H', h, self.fluid_string)

            # 2. Transport
            # Handle potential failures near critical point or two-phase
            try:
                visc = CP.PropsSI('V', 'P', p, 'H', h, self.fluid_string)
                cond = CP.PropsSI('L', 'P', p, 'H', h, self.fluid_string)
                cp = CP.PropsSI('Cpmass', 'P', p, 'H', h, self.fluid_string)
            except ValueError:
                # Fallback or specific two-phase handling could go here
                # For now, propagate error but log context
                raise ValueError(f"Transport property failure at P={p:.0f}, H={h:.0f}")

            # 3. Quality (Phase check)
            try:
                q = CP.PropsSI('Q', 'P', p, 'H', h, self.fluid_string)
            except:
                q = -1.0  # Standard for single phase

            prandtl = (cp * visc) / cond if cond > 0 else 0

            return FluidState(
                pressure=p,
                temperature=T,
                enthalpy=h,
                density=rho,
                specific_heat=cp,
                viscosity=visc,
                conductivity=cond,
                prandtl=prandtl,
                quality=q
            )

        except ValueError as e:
            # If P-H fails, it's often due to being out of bounds.
            raise ValueError(
                f"CoolProp P-H Lookup failed for {self.fluid_string} at P={p / 1e5:.2f} bar, H={h:.2f}. {e}")

    def _get_state_from_pt(self, p: float, t: float) -> FluidState:
        try:
            h = CP.PropsSI('H', 'P', p, 'T', t, self.fluid_string)
            rho = CP.PropsSI('D', 'P', p, 'T', t, self.fluid_string)
            visc = CP.PropsSI('V', 'P', p, 'T', t, self.fluid_string)
            cond = CP.PropsSI('L', 'P', p, 'T', t, self.fluid_string)
            cp = CP.PropsSI('Cpmass', 'P', p, 'T', t, self.fluid_string)

            try:
                q = CP.PropsSI('Q', 'P', p, 'T', t, self.fluid_string)
            except:
                q = -1.0

            prandtl = (cp * visc) / cond if cond > 0 else 0

            return FluidState(
                pressure=p,
                temperature=t,
                enthalpy=h,
                density=rho,
                specific_heat=cp,
                viscosity=visc,
                conductivity=cond,
                prandtl=prandtl,
                quality=q
            )
        except ValueError as e:
            raise ValueError(
                f"CoolProp P-T Lookup failed for {self.fluid_string} at P={p / 1e5:.2f} bar, T={t:.2f} K. {e}")

    def get_density(self, p: float, t: float) -> float:
        """Quick density lookup helper [kg/m^3]"""
        return CP.PropsSI('D', 'P', p, 'T', t, self.fluid_string)

    def get_saturation_temp(self, p: float) -> float:
        """Boiling point at given pressure [K]"""
        try:
            return CP.PropsSI('T', 'P', p, 'Q', 0, self.fluid_string)
        except:
            return None  # Supercritical


# --- Example Usage ---
if __name__ == "__main__":
    # Test Nitrous
    ox = Propellant("REFPROP::N2O")
    state_ox = ox.get_state(pressure=50e5, temperature=280)
    print(f"Nitrous @ 50bar/280K: Rho={state_ox.density:.1f}, Cp={state_ox.specific_heat:.1f}")

    # Test Ethanol mixture (using the mole fraction alias)
    fuel = Propellant("Ethanol90")
    state_fuel = fuel.get_state(pressure=50e5, temperature=300)
    print(f"Ethanol90 @ 50bar/300K: Rho={state_fuel.density:.1f}, Visc={state_fuel.viscosity:.2e}")