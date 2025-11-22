import math
import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj as CEA_Obj_units
from rocketcea.cea_obj import CEA_Obj, add_new_fuel, add_new_oxidizer, add_new_propellant
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from bisect import bisect_left
import CoolProp.CoolProp as CP
from scipy.optimize import fsolve
from .compr_flow import CFTools
#from .tc_contour import *
from .ChamberContour import *


# Main Engine Geometry and Combustion Analysis Class
class Thrust_Chamber:
    def __init__(self, id="Engine 1", fuel="Ethanol", ox="N2O", thrust=1500, p_c=50, MR=0,
                 L_star=1600, CR=15, ER=0, R_t=0, eta_c=0.95, eta_f=0.98):
        # if no MR, ER or D_t given, optimal values will be used
        self.id = id
        self.fuel = fuel
        self.ox = ox
        self.F = thrust
        self.p_c = p_c
        self.MR = MR
        self.L_star = L_star
        self.CR = CR
        self.ER = ER
        self.R_c = 0
        self.R_t = R_t
        self.R_e = 0
        self.A_c = 0
        self.A_t = 0
        self.A_e = 0
        self.l_percent = 80
        self.massflow = 0
        self.massflow_ox = 0
        self.massflow_f = 0
        self.Isp_ideal = 0
        self.Isp = 0
        self.Isp_vac = 0
        self.c_star = 0
        self.c_f = 0
        self.c_f_opt = 0
        self.c_star_opt = 0
        self.gamma = 0
        self.MW_c = 0
        self.T_c = 0
        self.eta_c = eta_c
        self.eta_f = eta_f
        self.tc_contourx = []
        self.tc_contoury = []
        self.tc_contoury_o = []
        self.tc_wall_t = 3
        self.t_pos = 0
        self.T_vec = []
        self.p_vec = []
        self.M_vec = []

        self.AR_vec = np.empty(0)
        self.Pr_c = 0
        self.Cp_c = 0
        self.mu_c = 0
        self.q = 0
        self.T_w = 700
        self.T_w_0 = 295
        self.k_w = 380
        self.CEA = 0
        self.CEA_output = ""
        self.pa = 1.013
        self.g = 9.81
        self.R_molar = 8314.5
        print("New Thrust Chamber Initialized!")

    def setup(self):
        # (Existing setup method unchanged, omitted for brevity)
        print("--- Thrust chamber analysis for: ", self.id)
        print("Fuel: ", self.fuel)
        print("Oxidizer: ", self.ox)
        print("Thrust: ", self.F)
        print("Chamber pressure: ", self.p_c)

        card_str = """
        fuel C2H5OH(L)   C 2 H 6 O 1
        h,cal=-66370.0      t(k)=500       wt%=90.0
        fuel water H 2.0 O 1.0  wt%=10.0
        h,cal=-68308.  t(k)=298 rho,g/cc = 0.9998
        """
        add_new_fuel('Ethanol90', card_str)

        card_str = """
        fuel C2H5OH(L)   C 2 H 6 O 1
        h,cal=-66370.0      t(k)=500       wt%=80.0
        fuel water H 2.0 O 1.0  wt%=20.0
        h,cal=-68308.  t(k)=298 rho,g/cc = 0.9998
        """
        add_new_fuel('Ethanol80', card_str)

        self.CEA = CEA_Obj_units(oxName=self.ox, fuelName=self.fuel, pressure_units='Bar', cstar_units='m/s', specific_heat_units='kJ/kg-K',
                              temperature_units='K', enthalpy_units='kJ/kg')
        self.cea_imp = CEA_Obj(oxName=self.ox, fuelName=self.fuel)

        compr_tools = CFTools()

        MR_vec = np.linspace(1, 10, 1000)
        c_star_vec = np.zeros(len(MR_vec))
        for x in range(len(MR_vec)):
            c_star_vec[x] = self.CEA.get_Cstar(Pc=self.p_c, MR=MR_vec[x])
        MR_opt_cstar = float(MR_vec[np.argmax(c_star_vec)])
        print("C* optimized MR: ", MR_opt_cstar)

        self.gamma = self.CEA.get_Chamber_MolWt_gamma(Pc=self.p_c, MR=MR_opt_cstar if self.MR == 0 else self.MR)[1]

        if self.MR == 0:
            temp = 0
            while ((self.gamma - temp) / self.gamma > 0.01):
                temp = self.gamma
                isp_vec = np.zeros(len(MR_vec))
                for x in range(len(MR_vec)):
                    isp_vec[x] = self.CEA.estimate_Ambient_Isp(Pc=self.p_c, MR=MR_vec[x], eps=compr_tools.get_ER_from_exitPressure(self.pa, self.p_c, self.gamma) if self.ER == 0 else self.ER, Pamb=1)[0]
                MR_opt_isp = float(MR_vec[np.argmax(isp_vec)])
                self.gamma = self.CEA.get_Chamber_MolWt_gamma(Pc=self.p_c, MR=MR_opt_isp)[1]
            self.MR = MR_opt_isp

        if self.ER == 0:
            self.ER = compr_tools.get_ER_from_exitPressure(self.pa, self.p_c, self.gamma)

        print("Expansion Ratio\t\t", self.ER)
        print("Gamma\t\t", self.gamma)
        print("ISP optimized MR\t\t", self.MR)

        self.c_star_opt = self.CEA.get_Cstar(Pc=self.p_c, MR=self.MR)
        self.c_star = self.c_star_opt * self.eta_c
        print("Ideal C*: ", self.c_star)

        self.c_f_opt = self.CEA.get_PambCf(Pc=self.p_c, MR=self.MR, eps=self.ER, Pamb=self.pa)[0]
        print(self.c_f_opt)
        self.c_f = self.c_f_opt * self.eta_f
        print("Ideal C_f: ", self.c_f_opt)

        self.Isp_vac = self.CEA.get_Isp(Pc=self.p_c, MR=self.MR, eps=self.ER)
        print("VAC ISP: ", self.Isp_vac)

        self.MW_c = self.CEA.get_Chamber_MolWt_gamma(Pc=self.p_c, MR=self.MR)[0]
        self.T_c = self.CEA.get_Temperatures(Pc=self.p_c, MR=self.MR, eps=None)[0]
        print("Tc: ", self.T_c)

        self.cea_output = self.cea_imp.get_full_cea_output(Pc=self.p_c, MR=self.MR, PcOvPe=self.p_c, short_output=1, show_transport=1, pc_units='bar', eps=None, output='siunits')

        self.Isp_ideal = self.CEA.estimate_Ambient_Isp(Pc=self.p_c, MR=self.MR, eps=self.ER, Pamb=self.pa)[0]
        self.Isp = self.Isp_ideal * self.eta_c * self.eta_f
        print("Ideal Isp: ", self.Isp_ideal)
        print("Real Isp:", self.Isp)

        self.massflow = self.F / (self.Isp * self.g)
        self.massflow_ox = self.massflow * (self.MR / (1 + self.MR))
        self.massflow_f = self.massflow - self.massflow_ox
        print("Propellant Massflow: ", self.massflow)

        if self.R_t == 0:
            self.A_t = (self.massflow / (self.p_c * 101325) * np.sqrt(self.R_molar * self.T_c / (self.MW_c * self.gamma * (2 / (self.gamma + 1)) ** ((self.gamma + 1) / (self.gamma - 1))))) * 1e6
            self.R_t = np.sqrt(self.A_t / np.pi)
        else:
            self.A_t = np.pi * self.R_t ** 2

        self.A_e = self.ER * self.A_t
        self.R_e = np.sqrt(self.A_e / np.pi)
        self.A_c = self.CR * self.A_t
        self.R_c = np.sqrt(self.A_c / np.pi)

        self.tc_contourx, self.tc_contoury = generate_tc_contour(self.ER, self.R_t, self.l_percent, self.CR, self.L_star)
        self.tc_contoury_o = self.tc_contoury + self.tc_wall_t
        print("Generated chamber contour")

        self.M_vec = np.zeros_like(self.tc_contourx)
        self.AR_vec = np.pi * self.tc_contoury ** 2 / self.A_t

        for x in range(len(self.M_vec)):
            if self.tc_contourx[x] <= self.t_pos:
                self.M_vec[x] = compr_tools.solve_AreaMachEquation(0, 1, self.AR_vec[x], self.gamma)
            elif self.tc_contoury[x] > self.t_pos:
                self.M_vec[x] = compr_tools.solve_AreaMachEquation(1, 4, self.AR_vec[x], self.gamma)

        self.p_vec = compr_tools.get_ThermodynamicConditions(self.M_vec, self.p_c, self.MW_c, self.T_c, self.gamma).get('p')
        self.T_vec = compr_tools.get_ThermodynamicConditions(self.M_vec, self.p_c, self.MW_c, self.T_c, self.gamma).get('T')

        print("Chamber Dimensions:")
        print("R_c: ", self.R_c)
        print("R_t: ", self.R_t)
        print("R_e: ", self.R_e)

        self.Pr_c = self.CEA.get_Chamber_Transport(Pc=self.p_c, MR=self.MR, eps=self.ER)[3]
        self.Cp_c = self.CEA.get_HeatCapacities(Pc=self.p_c, MR=self.MR, eps=self.ER)[0]
        self.mu_c = self.CEA.get_Chamber_Transport(Pc=self.p_c, MR=self.MR, eps=self.ER)[1]
        print("Chamber Transport Properties: ")
        print("Prandtl Number: ", self.Pr_c)
        print("Engine analysis finished!")



    def print_specification(self):
        # should print data in excel and generate spec sheet pdf
        pass

    def plot_chamber(self):
        plt.style.use('dark_background')
        figure, axis1 = plt.subplots()
        plt.grid(True, color="gray", linestyle='dotted')
        axis1.set_aspect('equal')
        axis1.plot(self.tc_contourx, self.tc_contoury, label='Inner Chamber Wall', color="sandybrown")
        axis1.plot(self.tc_contourx, -self.tc_contoury, color="peachpuff")
        axis1.set_xlabel('Axial length nozzle [mm]')
        axis1.set_ylabel('Y [mm]')
        axis1.set_ylim([0, self.tc_contoury[0] + 5])
        axis1.axvline(x=self.t_pos, linestyle='-.', color='gray', label='Throat Position')
        axis2 = axis1.twinx()
        axis2.set_ylabel('Mach number [-]')
        axis2.plot(self.tc_contourx, self.M_vec, label='Mach number', color='red', linestyle='-')
        line1, label1 = axis1.get_legend_handles_labels()
        line2, label2 = axis2.get_legend_handles_labels()
        lines = line1 + line2
        labels = label1 + label2
        axis1.legend(lines, labels, loc='upper left')
        plt.show()