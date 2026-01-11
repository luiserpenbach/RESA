"""
NASA CEA (Chemical Equilibrium with Applications) interface.

Wraps RocketCEA for combustion calculations with Ethanol/N2O.
"""

from typing import Dict, Optional
import warnings

try:
    from rocketcea.cea_obj import CEA_Obj
    HAS_ROCKETCEA = True
except ImportError:
    HAS_ROCKETCEA = False
    warnings.warn("RocketCEA not found. Install with: pip install rocketcea")


# Physical constants
G0 = 9.80665  # Standard gravity (m/s^2)

# Heating values
ETHANOL_LHV = 26.8e6  # Lower Heating Value (J/kg)

# Typical design ranges (for validation)
TYPICAL_C_STAR_RANGE = (1200, 1800)  # m/s for Ethanol/N2O
TYPICAL_FLAME_TEMP_RANGE = (2500, 3200)  # K for Ethanol/N2O


def validate_range(value: float, range_tuple: tuple, name: str,
                   warn_only: bool = True) -> bool:
    """Validate that a value falls within expected range.

    Args:
        value: Value to check
        range_tuple: (min, max) tuple defining acceptable range
        name: Parameter name for error message
        warn_only: If True, print warning instead of raising exception

    Returns:
        True if in range, False otherwise

    Raises:
        ValueError: If out of range and warn_only=False
    """
    min_val, max_val = range_tuple

    if min_val <= value <= max_val:
        return True

    msg = f"{name} = {value:.2e} is outside typical range [{min_val:.2e}, {max_val:.2e}]"

    if warn_only:
        print(f"WARNING: {msg}")
        return False
    else:
        raise ValueError(msg)


class CEACalculator:
    """Wrapper for NASA CEA combustion calculations.

    Handles Ethanol/N2O chemistry and provides clean interface
    for igniter performance predictions.
    """

    def __init__(self):
        """Initialize CEA calculator with Ethanol/N2O propellants."""
        if not HAS_ROCKETCEA:
            raise ImportError(
                "RocketCEA is required. Install with: pip install rocketcea"
            )

        # Initialize CEA with ethanol and N2O
        # CEA fuel card for ethanol: C2H5OH(L)
        # CEA oxidizer card for nitrous oxide: N2O
        self.cea = CEA_Obj(
            oxName='N2O',
            fuelName='C2H5OH',
        )

    def get_combustion_properties(
        self,
        mixture_ratio: float,
        chamber_pressure_pa: float,
        expansion_ratio: float = 1.0,
        frozen: bool = False
    ) -> Dict[str, float]:
        """Calculate combustion properties using CEA.

        Args:
            mixture_ratio: O/F mass ratio
            chamber_pressure_pa: Chamber pressure (Pa)
            expansion_ratio: Nozzle expansion ratio (Ae/At), default=1.0
            frozen: If True, use frozen flow; if False, use equilibrium

        Returns:
            Dictionary with combustion properties:
            - T_chamber: Chamber temperature (K)
            - T_exit: Exit temperature (K)
            - c_star: Characteristic velocity (m/s)
            - gamma: Specific heat ratio at throat
            - gamma_exit: Specific heat ratio at exit
            - MW: Molecular weight (kg/kmol)
            - isp: Vacuum specific impulse (s)
            - isp_vac: Vacuum specific impulse (s) - same as isp
        """
        # Convert pressure to psia for CEA
        chamber_pressure_psia = chamber_pressure_pa / 6894.76

        # Get chamber conditions
        if frozen:
            # Frozen flow (chemistry frozen at throat)
            isp, cstar, tc = self.cea.getFrozen_IvacCstrTc(
                Pc=chamber_pressure_psia,
                MR=mixture_ratio,
                eps=expansion_ratio
            )
        else:
            # Equilibrium flow (shifting equilibrium)
            isp, cstar, tc = self.cea.get_IvacCstrTc(
                Pc=chamber_pressure_psia,
                MR=mixture_ratio,
                eps=expansion_ratio
            )

        # Get throat gamma and molecular weight
        # RocketCEA returns (mw, gamma) tuple
        mw, gamma = self.cea.get_Throat_MolWt_gamma(
            Pc=chamber_pressure_psia,
            MR=mixture_ratio,
            eps=expansion_ratio
        )

        # Get exit properties if expansion ratio > 1
        if expansion_ratio > 1.0:
            # Get exit gamma
            mw_exit, gamma_exit = self.cea.get_exit_MolWt_gamma(
                Pc=chamber_pressure_psia,
                MR=mixture_ratio,
                eps=expansion_ratio
            )
            # Get exit temperature
            t_exit = self.cea.get_Temperatures(
                Pc=chamber_pressure_psia,
                MR=mixture_ratio,
                eps=expansion_ratio,
                frozen=int(frozen)
            )[2]  # Returns (Tc, Tthroat, Texit)
        else:
            gamma_exit = gamma
            t_exit = tc

        # Note: isp from get_IvacCstrTc is already vacuum Isp
        # The 'Ivac' in the method name means "vacuum specific impulse"
        isp_vac = isp

        # Convert temperatures from Rankine to Kelvin
        T_chamber = tc * 5/9  # Rankine to Kelvin
        T_exit = t_exit * 5/9 if expansion_ratio > 1.0 else T_chamber

        # C* is already in ft/s, convert to m/s
        c_star_ms = cstar * 0.3048

        # Isp from get_IvacCstrTc is vacuum Isp (seconds)
        # The 'Ivac' in method name stands for "vacuum specific impulse"

        results = {
            'T_chamber': T_chamber,
            'T_exit': T_exit,
            'c_star': c_star_ms,
            'gamma': gamma,
            'gamma_exit': gamma_exit,
            'MW': mw,
            'isp': isp,  # Vacuum Isp
            'isp_vac': isp_vac,  # Also vacuum Isp (same value)
        }

        # Validate results are in reasonable range
        validate_range(c_star_ms, TYPICAL_C_STAR_RANGE, "C*")
        validate_range(T_chamber, TYPICAL_FLAME_TEMP_RANGE, "Flame Temperature")

        return results

    def calculate_chemical_power_lhv(
        self,
        fuel_mass_flow: float,
        lhv_fuel: float = ETHANOL_LHV
    ) -> float:
        """Calculate chemical energy release rate using Lower Heating Value.

        This is the standard method for calculating igniter heat power output.

        Args:
            fuel_mass_flow: Fuel mass flow rate (kg/s)
            lhv_fuel: Lower heating value of fuel (J/kg)
                     Default: 26.8 MJ/kg for ethanol (ETHANOL_LHV)

        Returns:
            Chemical power release (W)
        """
        return fuel_mass_flow * lhv_fuel

    def get_sea_level_isp(self, mixture_ratio: float,
                          chamber_pressure_pa: float,
                          expansion_ratio: float,
                          ambient_pressure_pa: float = 101325.0) -> float:
        """Calculate Isp at specified ambient pressure (e.g., sea level).

        Args:
            mixture_ratio: O/F mass ratio
            chamber_pressure_pa: Chamber pressure (Pa)
            expansion_ratio: Nozzle expansion ratio
            ambient_pressure_pa: Ambient pressure (Pa), default sea level

        Returns:
            Specific impulse at given ambient pressure (s)
        """
        chamber_pressure_psia = chamber_pressure_pa / 6894.76
        ambient_pressure_psia = ambient_pressure_pa / 6894.76

        try:
            # Use estimate_Ambient_Isp with non-zero ambient pressure
            isp_amb = self.cea.estimate_Ambient_Isp(
                Pc=chamber_pressure_psia,
                MR=mixture_ratio,
                eps=expansion_ratio,
                Pamb=ambient_pressure_psia
            )[0]
            return isp_amb
        except:
            # Fallback: use get_Isp method if available
            try:
                isp_amb = self.cea.get_Isp(
                    Pc=chamber_pressure_psia,
                    MR=mixture_ratio,
                    eps=expansion_ratio
                )
                return isp_amb
            except:
                # Last resort: return vacuum Isp as approximation
                props = self.get_combustion_properties(
                    mixture_ratio, chamber_pressure_pa, expansion_ratio
                )
                return props['isp']

    def sweep_mixture_ratio(
        self,
        mr_min: float,
        mr_max: float,
        chamber_pressure_pa: float,
        num_points: int = 20,
        expansion_ratio: float = 1.0
    ) -> Dict[str, list]:
        """Sweep mixture ratio and return performance curves.

        Args:
            mr_min: Minimum mixture ratio
            mr_max: Maximum mixture ratio
            chamber_pressure_pa: Chamber pressure (Pa)
            num_points: Number of points in sweep
            expansion_ratio: Nozzle expansion ratio

        Returns:
            Dictionary with lists of:
            - mixture_ratios
            - c_star values
            - temperatures
            - isp values
        """
        import numpy as np

        mixture_ratios = np.linspace(mr_min, mr_max, num_points)

        c_stars = []
        temperatures = []
        isps = []
        gammas = []

        for mr in mixture_ratios:
            try:
                props = self.get_combustion_properties(
                    mr, chamber_pressure_pa, expansion_ratio
                )
                c_stars.append(props['c_star'])
                temperatures.append(props['T_chamber'])
                isps.append(props['isp'])
                gammas.append(props['gamma'])
            except Exception as e:
                # Skip points that fail (e.g., out of valid range)
                print(f"Warning: Failed for MR={mr:.2f}: {e}")
                continue

        return {
            'mixture_ratios': mixture_ratios.tolist(),
            'c_star': c_stars,
            'temperature': temperatures,
            'isp': isps,
            'gamma': gammas,
        }


def estimate_heat_power(mass_flow_total: float, mixture_ratio: float,
                        chamber_pressure_pa: float = None,
                        cea_calc: Optional[CEACalculator] = None,
                        lhv_fuel: float = ETHANOL_LHV) -> float:
    """Estimate thermal power output from igniter using fuel LHV.

    Calculates chemical energy release rate based on fuel mass flow
    and Lower Heating Value. This is the standard method for igniters.

    Args:
        mass_flow_total: Total mass flow rate (kg/s)
        mixture_ratio: O/F mass ratio
        chamber_pressure_pa: Chamber pressure (Pa) - not used, kept for compatibility
        cea_calc: CEA calculator instance - not used, kept for compatibility
        lhv_fuel: Lower heating value of fuel (J/kg)
                 Default: ETHANOL_LHV = 26.8 MJ/kg

    Returns:
        Thermal power output (W)

    Example:
        For 50 g/s total flow at O/F=2.0:
        - Fuel flow = 50/(1+2) = 16.67 g/s
        - Power = 0.01667 kg/s x 26.8 MJ/kg = 447 kW
    """
    # Calculate fuel mass flow from mixture ratio
    fuel_mass_flow = mass_flow_total / (1 + mixture_ratio)

    # Heat power = fuel flow x LHV
    power_w = fuel_mass_flow * lhv_fuel

    return power_w
