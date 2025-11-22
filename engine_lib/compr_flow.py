"""
Compressible flow tools mainly for thrust chamber design
"""

class CFTools:
  def __init__(self):
    self.g = 9.81         # Gravity
    self.R_molar = 8314.5 # Gas constant


  def get_ER_from_exitPressure(self, p_e, p_c, gamma):		# Calculate optimal expansion ratio
    ER = 1/(((gamma+1)/2)**(1/(gamma-1)) * (p_e/p_c)**(1/gamma) * ( (gamma+1)/(gamma-1)*( 1 - (p_e/p_c)**((gamma-1)/gamma) ) )**0.5)
    return ER

  def solve_AreaMachEquation(self, lowM, highM, AR, gamma):
    error = 0.0001
    result_lowM = self.AreaMachEquation(lowM, AR, gamma)
    result_highM = self.AreaMachEquation(highM, AR, gamma)

    while abs(result_highM-result_lowM) > error and abs(result_highM) > error:

      midM = 0.5*(lowM+highM)

      result_lowM = self.AreaMachEquation(lowM, AR, gamma)
      result_highM = self.AreaMachEquation(highM, AR, gamma)
      result_midM = self.AreaMachEquation(midM, AR, gamma)

      if result_lowM*result_midM > 0:
        lowM = midM
      elif result_highM*result_midM > 0:
        highM = midM

    return midM

  def AreaMachEquation(self, M, AR, gamma):
    """
    Returns Area-Mach Equation
    """
    return (M*AR)-((2+(gamma-1)*M**2)/(gamma+1))**((gamma+1)/(2*(gamma-1)))


  def get_ThermodynamicConditions(self, M, p_c, Mw, T_c, gamma):
    """
    Solves for local gas properties as a function of local Mach Number
    """

    rho_c = (p_c*Mw)/(self.R_molar*1e-3*T_c) # Gas density in combustion chamber

    T = T_c/(1+(gamma-1)/2*M**2) # Total temperature equation
    p = p_c*1.01325e5/(1+(gamma-1)/2*M**2)**(gamma/(gamma-1)) # Total pressure equation
    rho = rho_c/(1+(gamma-1)/2*M**2)**(1/(gamma-1))

    return{'T': T,'p': p,'rho': rho}