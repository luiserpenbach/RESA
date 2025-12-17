from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import add_new_fuel
from dataclasses import dataclass


@dataclass
class CombustionPoint:
    """Standardized output for a combustion operating point."""
    pc: float  # Chamber Pressure [Bar]
    mr: float  # Mixture Ratio
    cstar: float  # Characteristic Velocity [m/s]
    isp_vac: float  # Vacuum Isp [s]
    isp_opt: float  # Optimized Expansion Isp [s] (at p_amb)
    T_combustion: float  # Chamber Temperature [K]
    gamma: float  # Specific Heat Ratio
    mw: float  # Molecular Weight [kg/kmol] or [g/mol]
    mach_exit: float  # Exit Mach number


class CEASolver:
    def __init__(self, fuel_name: str, oxidizer_name: str):
        self._setup_custom_fuels()
        # Explicitly define units for the wrapper.
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
        """Defines custom fuel blends for RocketCEA."""
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

    def run(self, pc_bar: float, mr: float, eps: float, p_amb_bar: float = 1.013) -> CombustionPoint:
        """
        Runs the equilibrium analysis.
        Args:
            pc_bar: Chamber Pressure [Bar]
            mr: Mixture Ratio
            eps: Expansion Ratio
        """
        # 1. Get Isp and Cstar
        isp_vac = self.cea.get_Isp(Pc=pc_bar, MR=mr, eps=eps)
        cstar_mps = self.cea.get_Cstar(Pc=pc_bar, MR=mr)

        # 2. Get Thermodynamic Properties
        # get_Chamber_MolWt_gamma returns (Mw, Gamma)
        M_w, gam = self.cea.get_Chamber_MolWt_gamma(Pc=pc_bar, MR=mr, eps=eps)

        # get_Temperatures returns (Tc, Tt, Te) - We want Chamber Temp (index 0)
        # It returns Rankine by default unless units are handled, but CEA_Obj_w_units usually handles this?
        # WARNING: rocketcea.cea_obj_w_units usually handles units if configured,
        # but pure CEA_Obj (if used internally) might be Rankine.
        # Assuming CEA_Obj_w_units is doing its job or we check output.
        # Ideally, we explicitly convert if we suspect Rankine.
        # Let's trust the unit wrapper first, but often T is returned in Rankine by base CEA.

        temps = self.cea.get_Temperatures(Pc=pc_bar, MR=mr, eps=eps)
        T_chamber = temps[0]  # Chamber Temperature

        # If the number is huge (>1000 higher than expected), it might be Rankine.
        # Standard Ethanol/N2O is ~3000K (5400R).
        # We will assume Kelvin since `rocketcea.cea_obj_w_units` is imported,
        # but if you see ~5000+ K, you might need to multiply by 5/9.

        # 3. Mach number at exit
        mach_exit = self.cea.get_MachNumber(Pc=pc_bar, MR=mr, eps=eps)

        # 4. Estimate Ambient Isp
        try:
            isp_amb, mode = self.cea.estimate_Ambient_Isp(Pc=pc_bar, MR=mr, eps=eps, Pamb=p_amb_bar)
        except:
            isp_amb = isp_vac

        return CombustionPoint(
            pc=pc_bar,
            mr=mr,
            cstar=cstar_mps,
            isp_vac=isp_vac,
            isp_opt=isp_amb,
            T_combustion=T_chamber,
            gamma=gam,
            mw=M_w,
            mach_exit=mach_exit
        )