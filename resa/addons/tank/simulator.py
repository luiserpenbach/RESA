"""
Tank depletion simulation for bi-propellant rocket engine.

Provides tank simulation classes with correct physics for two-phase
self-pressurizing systems (N2O) and single-phase pressurized tanks (Ethanol).

Key physics:
- N2O vapor is always at saturation (not tracked as separate state)
- Proper evaporative cooling
- Correct pressure calculations using Dalton's law
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple

import CoolProp.CoolProp as CP

from .config import TankConfig, PressurantConfig, PropellantConfig
from . import thermodynamics as thermo


class TwoPhaseNitrousTank:
    """
    Revised N2O tank model with correct physics.

    State variables (only 4):
    - m_liquid: liquid propellant mass
    - m_pressurant: pressurant gas mass in ullage
    - T_liquid: liquid temperature
    - T_ullage: ullage gas temperature

    Key physics:
    - N2O vapor is always at saturation (determined by T, not tracked)
    - Evaporative cooling when liquid vaporizes
    - Correct pressure from Dalton's law: P_total = P_N2 + P_sat(T)
    """

    def __init__(
        self,
        tank_config: TankConfig,
        pressurant_config: PressurantConfig,
        propellant_config: PropellantConfig
    ):
        self.tank = tank_config
        self.pressurant = pressurant_config
        self.propellant = propellant_config

        self.prop_fluid = propellant_config.fluid_name
        self.pres_fluid = pressurant_config.fluid_name

        # Storage for current state
        self.current_pressure = tank_config.initial_ullage_pressure
        self.current_liquid_mass = tank_config.initial_liquid_mass

    def get_saturation_pressure(self, temperature: float) -> float:
        """Get saturation pressure of propellant at given temperature."""
        return thermo.get_saturation_pressure(temperature, self.prop_fluid)

    def get_liquid_density(self, temperature: float) -> float:
        """Get saturated liquid density at given temperature."""
        return thermo.get_liquid_density(temperature, self.prop_fluid)

    def get_vapor_density(self, temperature: float) -> float:
        """Get saturated vapor density at given temperature."""
        return thermo.get_vapor_density(temperature, self.prop_fluid)

    def get_latent_heat(self, temperature: float) -> float:
        """Get latent heat of vaporization [J/kg]."""
        return thermo.get_latent_heat(temperature, self.prop_fluid)

    def get_liquid_cp(self, temperature: float) -> float:
        """Get liquid heat capacity [J/kg/K]."""
        return thermo.get_liquid_cp(temperature, self.prop_fluid)

    def calculate_heat_transfer(self, T_liquid: float) -> float:
        """
        Calculate heat transfer rate from ambient [W].

        Simplified: assume tank at average of liquid temp and ambient.
        """
        # Surface area (assume cylindrical tank)
        A_total = 2 * np.pi * (self.tank.volume / (np.pi * 2))**(2/3) * 3

        # Heat transfer (positive = heating from ambient)
        Q_in = self.tank.heat_transfer_coefficient * A_total * (
            self.tank.ambient_temperature - T_liquid
        )

        return Q_in

    def state_derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Calculate time derivatives of state variables.

        State vector:
        [0] m_liquid - liquid propellant mass (kg)
        [1] m_pressurant - pressurant gas mass in ullage (kg)
        [2] T_liquid - liquid temperature (K)
        [3] T_ullage - ullage gas temperature (K)
        """
        m_liquid, m_pressurant, T_liquid, T_ullage = state

        # Prevent negative masses
        if m_liquid < 0.01:
            m_liquid = 0.01
        if m_pressurant < 0:
            m_pressurant = 1e-6

        # === GEOMETRY ===
        rho_liquid = self.get_liquid_density(T_liquid)
        V_liquid = m_liquid / rho_liquid
        V_ullage = self.tank.volume - V_liquid

        if V_ullage <= 1e-6:
            V_ullage = 1e-6

        # === PRESSURE CALCULATION ===
        # N2O saturation pressure (from liquid temperature)
        P_sat_N2O = self.get_saturation_pressure(T_liquid)

        # N2 pressurant pressure (ideal gas)
        R_N2 = thermo.get_gas_constant(self.pres_fluid)
        P_N2 = m_pressurant * R_N2 * T_ullage / V_ullage

        # Total pressure (Dalton's law)
        P_ullage = P_N2 + P_sat_N2O

        # Store for external access
        self.current_pressure = P_ullage
        self.current_liquid_mass = m_liquid

        # === MASS FLOW RATES ===
        # Liquid withdrawal to engine
        m_dot_out = self.propellant.mass_flow_rate

        # Vaporization rate to maintain pressure equilibrium
        # As liquid is withdrawn, ullage expands -> pressure drops -> liquid vaporizes
        # Rate of ullage volume expansion
        dV_ullage_dt = (m_dot_out / rho_liquid)

        # N2O vapor density in ullage (saturated)
        rho_vapor = self.get_vapor_density(T_liquid)

        # Mass of vapor needed to fill expanding ullage at saturation density
        # This is the evaporation rate
        m_dot_evap = rho_vapor * dV_ullage_dt

        # Total rate of liquid mass change
        dm_liquid_dt = -m_dot_out - m_dot_evap

        # Pressurant addition (proportional control)
        P_setpoint = self.pressurant.supply_pressure
        if P_ullage < P_setpoint:
            dm_pressurant_dt = self.pressurant.regulator_flow_coefficient * \
                              (P_setpoint - P_ullage)
            dm_pressurant_dt = min(dm_pressurant_dt, 0.01)  # Limit flow rate
        else:
            dm_pressurant_dt = 0

        # === ENERGY BALANCE ===
        # Heat transfer from ambient
        Q_ambient = self.calculate_heat_transfer(T_liquid)

        # Latent heat of vaporization
        h_fg = self.get_latent_heat(T_liquid)

        # Liquid heat capacity
        cp_liquid = self.get_liquid_cp(T_liquid)

        # Liquid temperature change
        # Energy in from ambient - energy out for vaporization
        if m_liquid > 0.01:
            dT_liquid_dt = (Q_ambient - m_dot_evap * h_fg) / (m_liquid * cp_liquid)
        else:
            dT_liquid_dt = 0

        # Ullage temperature (simplified - follows liquid with some lag)
        # In reality, ullage is cooled by expansion but heated by pressurant addition
        tau_thermal = 10.0  # Time constant for thermal equilibration (s)
        dT_ullage_dt = (T_liquid - T_ullage) / tau_thermal

        # Add heating from pressurant addition (if significant)
        if m_pressurant > 1e-6 and dm_pressurant_dt > 0:
            cp_N2 = 1040.0  # J/kg/K
            dT_ullage_dt += dm_pressurant_dt * cp_N2 * \
                           (self.pressurant.supply_temperature - T_ullage) / \
                           (m_pressurant * cp_N2)

        return np.array([
            dm_liquid_dt,
            dm_pressurant_dt,
            dT_liquid_dt,
            dT_ullage_dt
        ])

    def simulate(self, t_span: Tuple[float, float], dt: float = 0.1):
        """
        Run tank depletion simulation.

        Args:
            t_span: (t_start, t_end) in seconds
            dt: time step for output

        Returns:
            Solution object from solve_ivp
        """
        # === INITIAL STATE ===
        T_init = self.tank.initial_temperature
        P_target = self.tank.initial_ullage_pressure

        # Calculate initial pressurant mass needed
        V_liquid_init = self.tank.initial_liquid_mass / \
                       self.get_liquid_density(T_init)
        V_ullage_init = self.tank.volume - V_liquid_init

        # N2O provides self-pressurization
        P_sat_N2O = self.get_saturation_pressure(T_init)

        # Remaining pressure must come from N2
        P_N2_needed = P_target - P_sat_N2O

        if P_N2_needed < 0:
            print(f"Warning: N2O self-pressurization ({P_sat_N2O/1e5:.1f} bar) " +
                  f"exceeds target ({P_target/1e5:.1f} bar)")
            P_N2_needed = 0

        # Calculate N2 mass needed
        R_N2 = thermo.get_gas_constant(self.pres_fluid)
        m_pressurant_init = P_N2_needed * V_ullage_init / (R_N2 * T_init)

        print(f"Initial conditions:")
        print(f"  Liquid mass: {self.tank.initial_liquid_mass:.1f} kg")
        print(f"  Ullage volume: {V_ullage_init*1000:.1f} L")
        print(f"  N2O sat pressure: {P_sat_N2O/1e5:.1f} bar")
        print(f"  N2 pressure: {P_N2_needed/1e5:.1f} bar")
        print(f"  N2 mass needed: {m_pressurant_init*1000:.1f} g")
        print(f"  Total pressure: {P_target/1e5:.1f} bar")

        state_0 = np.array([
            self.tank.initial_liquid_mass,  # m_liquid
            m_pressurant_init,  # m_pressurant
            T_init,  # T_liquid
            T_init,  # T_ullage
        ])

        # Event: tank empty
        def tank_empty(t, state):
            return state[0] - 0.1  # Stop when liquid mass < 0.1 kg

        tank_empty.terminal = True

        # Solve ODE system
        sol = solve_ivp(
            self.state_derivatives,
            t_span,
            state_0,
            method='LSODA',
            dense_output=True,
            events=tank_empty,
            max_step=dt
        )

        return sol


class EthanolTank:
    """
    Ethanol tank model (single-phase liquid, no self-pressurization).

    State variables:
    - m_liquid: liquid mass
    - m_pressurant: pressurant mass
    - T_liquid: temperature
    """

    def __init__(
        self,
        tank_config: TankConfig,
        pressurant_config: PressurantConfig,
        propellant_config: PropellantConfig
    ):
        self.tank = tank_config
        self.pressurant = pressurant_config
        self.propellant = propellant_config

        self.prop_fluid = propellant_config.fluid_name
        self.pres_fluid = pressurant_config.fluid_name

        self.current_pressure = tank_config.initial_ullage_pressure
        self.current_liquid_mass = tank_config.initial_liquid_mass

    def get_liquid_density(self, temperature: float) -> float:
        """Get liquid density."""
        try:
            return CP.PropsSI('D', 'T', temperature, 'P', 1e5, self.prop_fluid)
        except Exception:
            # Ethanol density with thermal expansion
            rho_20C = 789  # kg/m^3
            alpha = 1.1e-3  # 1/K thermal expansion
            return rho_20C * (1 - alpha * (temperature - 293.15))

    def state_derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Calculate state derivatives.

        State vector:
        [0] m_liquid - liquid mass (kg)
        [1] m_pressurant - pressurant mass (kg)
        [2] T_liquid - temperature (K)
        """
        m_liquid, m_pressurant, T_liquid = state

        if m_liquid < 0.01:
            m_liquid = 0.01
        if m_pressurant < 0:
            m_pressurant = 1e-6

        # Geometry
        rho_liquid = self.get_liquid_density(T_liquid)
        V_liquid = m_liquid / rho_liquid
        V_ullage = self.tank.volume - V_liquid

        if V_ullage <= 1e-6:
            V_ullage = 1e-6

        # Pressure (ideal gas)
        R_pres = thermo.get_gas_constant(self.pres_fluid)
        P_ullage = m_pressurant * R_pres * T_liquid / V_ullage

        self.current_pressure = P_ullage
        self.current_liquid_mass = m_liquid

        # Mass flow
        m_dot_out = self.propellant.mass_flow_rate
        dm_liquid_dt = -m_dot_out

        # Pressurant addition
        P_setpoint = self.pressurant.supply_pressure
        if P_ullage < P_setpoint:
            dm_pressurant_dt = self.pressurant.regulator_flow_coefficient * \
                              (P_setpoint - P_ullage)
            dm_pressurant_dt = min(dm_pressurant_dt, 0.01)
        else:
            dm_pressurant_dt = 0

        # Heat transfer
        A_total = 2 * np.pi * (self.tank.volume / (np.pi * 2))**(2/3) * 3
        Q_in = self.tank.heat_transfer_coefficient * A_total * \
               (self.tank.ambient_temperature - T_liquid)

        # Temperature change
        try:
            cp_liquid = CP.PropsSI('C', 'T', T_liquid, 'P', P_ullage,
                                  self.prop_fluid)
        except Exception:
            cp_liquid = 2400  # J/kg/K for ethanol

        if m_liquid > 0.01:
            dT_liquid_dt = Q_in / (m_liquid * cp_liquid)
        else:
            dT_liquid_dt = 0

        return np.array([dm_liquid_dt, dm_pressurant_dt, dT_liquid_dt])

    def simulate(self, t_span: Tuple[float, float], dt: float = 0.1):
        """
        Run simulation.

        Args:
            t_span: (t_start, t_end) in seconds
            dt: time step for output

        Returns:
            Solution object from solve_ivp
        """
        # Calculate initial pressurant mass
        T_init = self.tank.initial_temperature
        P_target = self.tank.initial_ullage_pressure

        rho_liquid = self.get_liquid_density(T_init)
        V_liquid_init = self.tank.initial_liquid_mass / rho_liquid
        V_ullage_init = self.tank.volume - V_liquid_init

        R_pres = thermo.get_gas_constant(self.pres_fluid)
        m_pressurant_init = P_target * V_ullage_init / (R_pres * T_init)

        print(f"Ethanol initial conditions:")
        print(f"  Liquid mass: {self.tank.initial_liquid_mass:.1f} kg")
        print(f"  Ullage volume: {V_ullage_init*1000:.1f} L")
        print(f"  N2 mass needed: {m_pressurant_init*1000:.1f} g")

        state_0 = np.array([
            self.tank.initial_liquid_mass,
            m_pressurant_init,
            T_init
        ])

        def tank_empty(t, state):
            return state[0] - 0.1

        tank_empty.terminal = True

        sol = solve_ivp(
            self.state_derivatives,
            t_span,
            state_0,
            method='LSODA',
            dense_output=True,
            events=tank_empty,
            max_step=dt
        )

        return sol
