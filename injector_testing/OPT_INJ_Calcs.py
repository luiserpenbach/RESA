import pandas as pd
import CoolProp.CoolProp as CP
import numpy as np


def populate_injector_calculations(df, area_map_m2):
    """
    Populates 'area', 'CALC_rho', 'CALC_cd', and 'CALC_reynolds' columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the flow test data.
    area_map_m2 : dict
        Dictionary mapping 'injector_sn' to Area in square meters (m^2).
    """

    def calculate_row(row):
        # 1. Extract Inputs & Handle Units
        sn = row.get('injector_sn')
        p_up_bar = row.get('p_up_avg')  # Bar
        t_up_c = row.get('T_up_avg')  # K
        mf_g_s = row.get('mf_avg')  # g/s
        fluid = row.get('fluid', 'N2')  # Default to N2
        is_choked = row.get('is_choked', True)

        # 2. Get Area from Dictionary
        if sn not in area_map_m2:
            return pd.Series([np.nan, np.nan, np.nan, np.nan])

        area = area_map_m2[sn]

        # 3. Standardize Units to SI (Pa, K, kg/s)
        p_up_pa = p_up_bar * 1e5
        t_up_k = t_up_c + 273.15
        mf_kg_s = mf_g_s / 1000.0

        # 4. Calculate Density (Rho) and Viscosity (Mu)
        rho = CP.PropsSI('D', 'P', p_up_pa, 'T', t_up_k, fluid)
        mu = CP.PropsSI('V', 'P', p_up_pa, 'T', t_up_k, fluid)
        gamma = CP.PropsSI('Cpmass', 'P', p_up_pa, 'T', t_up_k, fluid) / \
                CP.PropsSI('Cvmass', 'P', p_up_pa, 'T', t_up_k, fluid)
        gas_constant = CP.PropsSI('GAS_CONSTANT', fluid) / CP.PropsSI('MOLAR_MASS', fluid)

        # 5. Calculate Cd (Coefficient of Discharge)
        # Using Isentropic Choked Flow Equation for Gas
        if is_choked:
            # Calculate Theoretical Mass Flow (Ideal)
            # m_dot = A * P * sqrt(gamma/RT) * (2/(gamma+1))^((gamma+1)/(2(gamma-1)))
            term1 = np.sqrt(gamma / (gas_constant * t_up_k))
            term2 = (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
            psi = term1 * term2

            theoretical_mf = area * p_up_pa * psi

            cd = mf_kg_s / theoretical_mf if theoretical_mf > 0 else np.nan
        else:
            # Unchoked/Incompressible approximation (if needed)
            p_down_pa = row.get('p_down_avg', 1.01) * 1e5
            delta_p = p_up_pa - p_down_pa
            theoretical_mf = area * np.sqrt(2 * rho * delta_p)
            cd = mf_kg_s / theoretical_mf if theoretical_mf > 0 else np.nan

        # 6. Calculate Reynolds Number
        # Re = (rho * v * D) / mu  => Re = (4 * m_dot) / (pi * D * mu)
        # Effective Diameter
        d_eff = np.sqrt(4 * area / np.pi)

        if mu > 0:
            reynolds = (4 * mf_kg_s) / (np.pi * d_eff * mu)
        else:
            reynolds = np.nan

        return pd.Series([area, rho, cd, reynolds])

    # Apply the function to the DataFrame
    result_cols = ['area', 'CALC_rho', 'CALC_cd', 'CALC_reynolds']
    df[result_cols] = df.apply(calculate_row, axis=1)
    print("Populated injector flow parameters")

    return df

if __name__ == '__main__':
    opt_table = pd.read_csv("local_data/OPT-INJ_LCSC_B1_CF.csv")
    area_ideal = 3*0.41*1e-6 # 0.7
    area_up = 3*0.537*1e-6 #0.8
    print(area_up/area_ideal)

    demo_area_map = {
        'LCSC-5-35_Fuel': area_up,
        'LCSC-5-36_Fuel': 3*0.435*1e-6,
        'LCSC-5-35_Ox': 20.773e-6,
        'LCSC-5-36_Ox': 20.773e-6,
        'LCSC-5-37_Ox': 20.773e-6
    }


    opt_table_ext = populate_injector_calculations(opt_table, demo_area_map)

    # 4. View or Save
    print(opt_table_ext[['injector_sn', 'p_up_avg', 'mf_avg', 'CALC_cd', 'CALC_reynolds']].head())
    opt_table_ext.to_csv("local_data/OPT-INJ_LCSC_B1_CF_Calculated.csv", index=False)
