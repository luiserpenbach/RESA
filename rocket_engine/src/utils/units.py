class Units:
    """
    Explicit Unit Conversion Factors and Physical Constants.
    Usage:
        value_SI = value_Imperial * Units.INCH_TO_METER
        force = mass * Units.g0
    """

    # --- Physical Constants (SI) ---
    g0 = 9.80665  # Standard Gravity [m/s^2]
    R_univ = 8314.462618  # Universal Gas Constant [J/(kmol*K)]
    sigma_SB = 5.670374e-8  # Stefan-Boltzmann constant [W/(m^2*K^4)]

    # --- Length ---
    INCH_TO_METER = 0.0254
    METER_TO_INCH = 1.0 / INCH_TO_METER

    FEET_TO_METER = 0.3048
    METER_TO_FEET = 1.0 / FEET_TO_METER

    MM_TO_METER = 1e-3
    METER_TO_MM = 1e3

    # --- Mass ---
    LB_TO_KG = 0.45359237
    KG_TO_LB = 1.0 / LB_TO_KG

    SLUG_TO_KG = 14.5939
    KG_TO_SLUG = 1.0 / SLUG_TO_KG

    # --- Force ---
    LBF_TO_NEWTON = 4.4482216152605
    NEWTON_TO_LBF = 1.0 / LBF_TO_NEWTON

    # --- Pressure ---
    # Base is Pascal [Pa]
    PSI_TO_PA = 6894.757293168
    PA_TO_PSI = 1.0 / PSI_TO_PA

    BAR_TO_PA = 1e5
    PA_TO_BAR = 1e-5

    ATM_TO_PA = 101325.0
    PA_TO_ATM = 1.0 / ATM_TO_PA

    # --- Energy & Power ---
    BTU_TO_JOULE = 1055.05585
    JOULE_TO_BTU = 1.0 / BTU_TO_JOULE

    HP_TO_WATT = 745.699872
    WATT_TO_HP = 1.0 / HP_TO_WATT

    # --- Viscosity ---
    # Dynamic Viscosity [Pa*s] vs Poise
    POISE_TO_PAS = 0.1
    PAS_TO_POISE = 10.0
    CENTIPOISE_TO_PAS = 1e-3

    # --- Temperature Helper Methods ---
    @staticmethod
    def fahrenheit_to_kelvin(F: float) -> float:
        return (F + 459.67) * 5.0 / 9.0

    @staticmethod
    def kelvin_to_fahrenheit(K: float) -> float:
        return (K * 9.0 / 5.0) - 459.67

    @staticmethod
    def rankine_to_kelvin(R: float) -> float:
        return R * 5.0 / 9.0

    @staticmethod
    def kelvin_to_rankine(K: float) -> float:
        return K * 9.0 / 5.0


# --- Bartz Equation Specific Legacy Helpers ---
def bartz_sigma_factor(Tw_gas_side: float, T_stagnation: float, gamma: float, Mach: float) -> float:
    """
    Calculates the Bartz boundary layer correction factor (sigma).
    All inputs in SI (Kelvin).
    """
    # Stagnation to Static ratio term
    stag_ratio = 1 + 0.5 * (gamma - 1) * Mach ** 2

    # Simple Bartz approximation for sizing
    # Sigma = 1 / [ (0.5 * Tw/T0 * (1 + (g-1)/2 M^2) + 0.5)^0.68 * (1 + (g-1)/2 M^2)^0.12 ]

    # Typical wall to gas temp ratio guess
    wall_ratio = 0.6

    term1 = 0.5 * wall_ratio * stag_ratio + 0.5
    sigma = (term1 ** 0.68 * stag_ratio ** 0.12) ** -1.0
    return sigma