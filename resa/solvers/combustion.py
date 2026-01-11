"""CEA Combustion Solver for RESA."""
from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import add_new_fuel

from resa.core.results import CombustionResult


class CEASolver:
    """
    NASA CEA wrapper for combustion analysis.

    Provides equilibrium combustion calculations for liquid rocket engines.
    """

    def __init__(self, fuel_name: str, oxidizer_name: str):
        """
        Initialize CEA solver with propellant combination.

        Args:
            fuel_name: Fuel name (e.g., 'Ethanol90', 'RP-1')
            oxidizer_name: Oxidizer name (e.g., 'N2O', 'LOX')
        """
        self._setup_custom_fuels()
        self.cea = CEA_Obj(
            oxName=oxidizer_name,
            fuelName=fuel_name,
            cstar_units='m/s',
            pressure_units='Bar',
            temperature_units='K',
            isp_units='sec',
            specific_heat_units='J/kg-K'
        )

    def _setup_custom_fuels(self):
        """Define custom fuel blends for RocketCEA."""
        # 90% Ethanol / 10% Water
        card_str_90 = """
        fuel C2H5OH(L)   C 2 H 6 O 1
        h,cal=-66370.0      t(k)=298.15       wt%=90.0
        fuel water H 2.0 O 1.0  wt%=10.0
        h,cal=-68308.  t(k)=298.15 rho,g/cc = 0.9998
        """
        add_new_fuel('Ethanol90', card_str_90)

        # 80% Ethanol / 20% Water
        card_str_80 = """
        fuel C2H5OH(L)   C 2 H 6 O 1
        h,cal=-66370.0      t(k)=298.15       wt%=80.0
        fuel water H 2.0 O 1.0  wt%=20.0
        h,cal=-68308.  t(k)=298.15 rho,g/cc = 0.9998
        """
        add_new_fuel('Ethanol80', card_str_80)

    def run(self, pc_bar: float, mr: float, eps: float, p_amb_bar: float = 1.013) -> CombustionResult:
        """
        Run equilibrium combustion analysis.

        Args:
            pc_bar: Chamber pressure [bar]
            mr: Mixture ratio (O/F)
            eps: Expansion ratio
            p_amb_bar: Ambient pressure [bar] for sea-level Isp

        Returns:
            CombustionResult with all thermodynamic properties
        """
        # Get Isp and Cstar
        isp_vac = self.cea.get_Isp(Pc=pc_bar, MR=mr, eps=eps)
        cstar_mps = self.cea.get_Cstar(Pc=pc_bar, MR=mr)

        # Get thermodynamic properties
        M_w, gam = self.cea.get_Chamber_MolWt_gamma(Pc=pc_bar, MR=mr, eps=eps)

        # Get temperatures (Chamber, Throat, Exit)
        temps = self.cea.get_Temperatures(Pc=pc_bar, MR=mr, eps=eps)
        T_chamber = temps[0]

        # Exit Mach number
        mach_exit = self.cea.get_MachNumber(Pc=pc_bar, MR=mr, eps=eps)

        # Estimate ambient Isp
        try:
            isp_amb, mode = self.cea.estimate_Ambient_Isp(
                Pc=pc_bar, MR=mr, eps=eps, Pamb=p_amb_bar
            )
        except Exception:
            isp_amb = isp_vac

        return CombustionResult(
            pc_bar=pc_bar,
            mr=mr,
            cstar=cstar_mps,
            isp_vac=isp_vac,
            isp_opt=isp_amb,
            T_combustion=T_chamber,
            gamma=gam,
            mw=M_w,
            mach_exit=mach_exit
        )
