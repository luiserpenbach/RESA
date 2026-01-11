"""
Tank Depletion Simulation for Bi-Propellant Rocket Engine - REVISED VERSION

Corrected physics for two-phase self-pressurizing systems:
- N2O vapor is always at saturation (not tracked as separate state)
- Proper evaporative cooling
- Correct pressure calculations

Author: Luis
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Tuple, Optional

# Try to import CoolProp, fall back to mock if not available

import CoolProp.CoolProp as CP


try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available - will use matplotlib")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available - plotting disabled")


@dataclass
class TankConfig:
    """Configuration for a single pressurized tank"""
    volume: float  # m^3
    initial_liquid_mass: float  # kg
    initial_ullage_pressure: float  # Pa
    initial_temperature: float  # K
    wall_material_properties: dict  # thermal properties
    ambient_temperature: float  # K
    heat_transfer_coefficient: float  # W/m^2/K


@dataclass
class PressurantConfig:
    """Configuration for pressurant gas system"""
    fluid_name: str  # e.g., 'Nitrogen'
    supply_pressure: float  # Pa (regulated pressure)
    supply_temperature: float  # K
    regulator_flow_coefficient: float  # kg/s/Pa for pressure control


@dataclass
class PropellantConfig:
    """Configuration for propellant properties"""
    fluid_name: str  # CoolProp fluid name
    mass_flow_rate: float  # kg/s to engine
    is_self_pressurizing: bool  # True for N2O


class TwoPhaseNitrousTank:
    """
    Revised N2O tank model with correct physics:

    State variables (only 4 now):
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
        """Get saturation pressure of propellant at given temperature"""
        try:
            return CP.PropsSI('P', 'T', temperature, 'Q', 0, self.prop_fluid)
        except:
            return CP.PropsSI('Pcrit', self.prop_fluid)

    def get_liquid_density(self, temperature: float) -> float:
        """Get saturated liquid density at given temperature"""
        try:
            return CP.PropsSI('D', 'T', temperature, 'Q', 0, self.prop_fluid)
        except:
            # Fallback to compressed liquid at high pressure
            P_sat = self.get_saturation_pressure(temperature)
            return CP.PropsSI('D', 'P', P_sat, 'T', temperature, self.prop_fluid)

    def get_vapor_density(self, temperature: float) -> float:
        """Get saturated vapor density at given temperature"""
        try:
            return CP.PropsSI('D', 'T', temperature, 'Q', 1, self.prop_fluid)
        except:
            return 20.0  # Fallback ~20 kg/m³ at room temp

    def get_latent_heat(self, temperature: float) -> float:
        """Get latent heat of vaporization [J/kg]"""
        h_vapor = CP.PropsSI('H', 'T', temperature, 'Q', 1, self.prop_fluid)
        h_liquid = CP.PropsSI('H', 'T', temperature, 'Q', 0, self.prop_fluid)
        return h_vapor - h_liquid

    def get_liquid_cp(self, temperature: float) -> float:
        """Get liquid heat capacity [J/kg/K]"""
        try:
            return CP.PropsSI('C', 'T', temperature, 'Q', 0, self.prop_fluid)
        except:
            return 2000.0  # J/kg/K default

    def calculate_heat_transfer(self, T_liquid: float) -> float:
        """
        Calculate heat transfer rate from ambient [W]
        Simplified: assume tank at average of liquid temp and ambient
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
        Calculate time derivatives of state variables

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
        R_N2 = CP.PropsSI('gas_constant', self.pres_fluid) / \
               CP.PropsSI('molar_mass', self.pres_fluid)
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
        # As liquid is withdrawn, ullage expands → pressure drops → liquid vaporizes
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
        Run tank depletion simulation

        Args:
            t_span: (t_start, t_end) in seconds
            dt: time step for output
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
        R_N2 = CP.PropsSI('gas_constant', self.pres_fluid) / \
               CP.PropsSI('molar_mass', self.pres_fluid)
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
    Ethanol tank model (single-phase liquid, no self-pressurization)

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
        """Get liquid density"""
        try:
            return CP.PropsSI('D', 'T', temperature, 'P', 1e5, self.prop_fluid)
        except:
            # Ethanol density with thermal expansion
            rho_20C = 789  # kg/m³
            alpha = 1.1e-3  # 1/K thermal expansion
            return rho_20C * (1 - alpha * (temperature - 293.15))

    def state_derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
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
        R_pres = CP.PropsSI('gas_constant', self.pres_fluid) / \
                 CP.PropsSI('molar_mass', self.pres_fluid)
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
        except:
            cp_liquid = 2400  # J/kg/K for ethanol

        if m_liquid > 0.01:
            dT_liquid_dt = Q_in / (m_liquid * cp_liquid)
        else:
            dT_liquid_dt = 0

        return np.array([dm_liquid_dt, dm_pressurant_dt, dT_liquid_dt])

    def simulate(self, t_span: Tuple[float, float], dt: float = 0.1):
        """Run simulation"""

        # Calculate initial pressurant mass
        T_init = self.tank.initial_temperature
        P_target = self.tank.initial_ullage_pressure

        rho_liquid = self.get_liquid_density(T_init)
        V_liquid_init = self.tank.initial_liquid_mass / rho_liquid
        V_ullage_init = self.tank.volume - V_liquid_init

        R_pres = CP.PropsSI('gas_constant', self.pres_fluid) / \
                 CP.PropsSI('molar_mass', self.pres_fluid)
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


def plot_n2o_tank_matplotlib(sol, tank_obj, output_path=None):
    """
    Create detailed plots of N2O tank behavior using matplotlib

    Args:
        sol: Solution object from solve_ivp
        tank_obj: TwoPhaseNitrousTank instance
        output_path: Path to save figure (optional)
    """

    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available - skipping plotting")
        return None

    import CoolProp as CP

    # Time array
    t = sol.t

    # State variables
    m_liquid = sol.y[0, :]
    m_pressurant = sol.y[1, :]
    T_liquid = sol.y[2, :]
    T_ullage = sol.y[3, :]

    # Calculate derived quantities
    pressures = []
    P_N2_array = []
    P_sat_array = []
    V_ullage_array = []
    m_vapor_array = []

    for i in range(len(t)):
        # Geometry
        rho_liquid = tank_obj.get_liquid_density(T_liquid[i])
        V_liquid = m_liquid[i] / rho_liquid
        V_ullage = tank_obj.tank.volume - V_liquid
        V_ullage_array.append(V_ullage * 1000)  # L

        # Pressures
        P_sat = tank_obj.get_saturation_pressure(T_liquid[i])
        R_N2 = CP.PropsSI('gas_constant', 'Nitrogen') / CP.PropsSI('molar_mass', 'Nitrogen')
        P_N2 = m_pressurant[i] * R_N2 * T_ullage[i] / V_ullage
        P_total = P_N2 + P_sat

        pressures.append(P_total / 1e5)  # bar
        P_N2_array.append(P_N2 / 1e5)
        P_sat_array.append(P_sat / 1e5)

        # Vapor mass
        rho_vapor = tank_obj.get_vapor_density(T_liquid[i])
        m_vapor = rho_vapor * V_ullage
        m_vapor_array.append(m_vapor)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'N2O Tank Depletion Analysis (Burn Duration: {t[-1]:.1f} s)',
                 fontsize=14, fontweight='bold')

    # Row 0, Col 0: Total Pressure
    ax = axes[0, 0]
    ax.plot(t, pressures, 'b-', linewidth=2, label='Total Pressure')
    ax.axhline(y=100, color='r', linestyle='--', linewidth=1.5, label='Target (100 bar)')
    ax.set_ylabel('Pressure (bar)', fontsize=11)
    ax.set_title('Total Tank Pressure', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Row 0, Col 1: Masses
    ax = axes[0, 1]
    ax.plot(t, m_liquid, 'b-', linewidth=2, label='Liquid N2O')
    ax.plot(t, m_vapor_array, 'c:', linewidth=2, label='Vapor N2O')
    ax.set_ylabel('Mass (kg)', fontsize=11)
    ax.set_title('Liquid Mass & Vapor Mass', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Row 1, Col 0: Pressure Components
    ax = axes[1, 0]
    ax.plot(t, P_N2_array, 'g-', linewidth=2, label='N2 Contribution')
    ax.plot(t, P_sat_array, color='orange', linewidth=2, label='N2O Self-Press.')
    ax.plot(t, pressures, 'b--', linewidth=1.5, alpha=0.7, label='Total')
    ax.set_ylabel('Pressure (bar)', fontsize=11)
    ax.set_title('Pressure Components', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Row 1, Col 1: Temperatures
    ax = axes[1, 1]
    ax.plot(t, T_liquid - 273.15, 'r-', linewidth=2, label='Liquid')
    ax.plot(t, T_ullage - 273.15, color='orange', linestyle=':',
            linewidth=2, label='Ullage')
    ax.axhline(y=tank_obj.tank.ambient_temperature - 273.15,
               color='gray', linestyle='--', linewidth=1, label='Ambient')
    ax.set_ylabel('Temperature (°C)', fontsize=11)
    ax.set_title('Temperature Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Row 2, Col 0: Pressurant Mass
    ax = axes[2, 0]
    ax.fill_between(t, 0, np.array(m_pressurant)*1000,
                     color='green', alpha=0.3)
    ax.plot(t, np.array(m_pressurant)*1000, 'g-', linewidth=2, label='N2 Mass')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Mass (g)', fontsize=11)
    ax.set_title('Pressurant Mass', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Row 2, Col 1: Ullage Volume
    ax = axes[2, 1]
    ax.fill_between(t, 0, V_ullage_array, color='purple', alpha=0.3)
    ax.plot(t, V_ullage_array, color='purple', linewidth=2, label='Ullage Volume')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Volume (L)', fontsize=11)
    ax.set_title('Ullage Volume', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to: {output_path}")

    return fig


def plot_n2o_tank_detailed(sol, tank_obj):
    """
    Create detailed plots of N2O tank behavior over time

    Args:
        sol: Solution object from solve_ivp
        tank_obj: TwoPhaseNitrousTank instance (to access properties)
    """

    if not PLOTLY_AVAILABLE:
        print("Plotly not available - skipping plotting")
        return None

    import CoolProp as CP

    # Time array
    t = sol.t

    # State variables
    m_liquid = sol.y[0, :]
    m_pressurant = sol.y[1, :]
    T_liquid = sol.y[2, :]
    T_ullage = sol.y[3, :]

    # Calculate derived quantities at each time point
    pressures = []
    P_N2_array = []
    P_sat_array = []
    V_ullage_array = []
    m_vapor_array = []

    for i in range(len(t)):
        # Geometry
        rho_liquid = tank_obj.get_liquid_density(T_liquid[i])
        V_liquid = m_liquid[i] / rho_liquid
        V_ullage = tank_obj.tank.volume - V_liquid
        V_ullage_array.append(V_ullage * 1000)  # Convert to L

        # Pressures
        P_sat = tank_obj.get_saturation_pressure(T_liquid[i])
        R_N2 = CP.PropsSI('gas_constant', 'Nitrogen') / CP.PropsSI('molar_mass', 'Nitrogen')
        P_N2 = m_pressurant[i] * R_N2 * T_ullage[i] / V_ullage
        P_total = P_N2 + P_sat

        pressures.append(P_total / 1e5)  # Convert to bar
        P_N2_array.append(P_N2 / 1e5)
        P_sat_array.append(P_sat / 1e5)

        # Vapor mass (at saturation)
        rho_vapor = tank_obj.get_vapor_density(T_liquid[i])
        m_vapor = rho_vapor * V_ullage
        m_vapor_array.append(m_vapor)

    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Total Tank Pressure', 'Liquid Mass & Vapor Mass',
            'Pressure Components', 'Temperature Evolution',
            'Pressurant Mass', 'Ullage Volume'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # Row 1, Col 1: Total Pressure
    fig.add_trace(
        go.Scatter(x=t, y=pressures, name='Total Pressure',
                   line=dict(color='darkblue', width=3),
                   mode='lines'),
        row=1, col=1
    )
    # Add target pressure line
    fig.add_trace(
        go.Scatter(x=[t[0], t[-1]], y=[100, 100],
                   name='Target (100 bar)',
                   line=dict(color='red', width=2, dash='dash'),
                   mode='lines'),
        row=1, col=1
    )

    # Row 1, Col 2: Masses
    fig.add_trace(
        go.Scatter(x=t, y=m_liquid, name='Liquid N2O',
                   line=dict(color='blue', width=2),
                   mode='lines'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=t, y=m_vapor_array, name='Vapor N2O',
                   line=dict(color='lightblue', width=2, dash='dot'),
                   mode='lines'),
        row=1, col=2
    )

    # Row 2, Col 1: Pressure Components
    fig.add_trace(
        go.Scatter(x=t, y=P_N2_array, name='N2 Contribution',
                   line=dict(color='green', width=2),
                   mode='lines'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=P_sat_array, name='N2O Self-Pressurization',
                   line=dict(color='orange', width=2),
                   mode='lines'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=pressures, name='Total',
                   line=dict(color='darkblue', width=2, dash='dash'),
                   mode='lines'),
        row=2, col=1
    )

    # Row 2, Col 2: Temperatures
    fig.add_trace(
        go.Scatter(x=t, y=T_liquid - 273.15, name='Liquid Temp',
                   line=dict(color='red', width=2),
                   mode='lines'),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=t, y=T_ullage - 273.15, name='Ullage Temp',
                   line=dict(color='orange', width=2, dash='dot'),
                   mode='lines'),
        row=2, col=2
    )
    # Add ambient temperature line
    T_ambient = tank_obj.tank.ambient_temperature
    fig.add_trace(
        go.Scatter(x=[t[0], t[-1]], y=[T_ambient-273.15, T_ambient-273.15],
                   name='Ambient',
                   line=dict(color='gray', width=1, dash='dash'),
                   mode='lines'),
        row=2, col=2
    )

    # Row 3, Col 1: Pressurant Mass
    fig.add_trace(
        go.Scatter(x=t, y=np.array(m_pressurant)*1000, name='N2 Mass',
                   line=dict(color='green', width=2),
                   mode='lines',
                   fill='tozeroy',
                   fillcolor='rgba(0, 255, 0, 0.1)'),
        row=3, col=1
    )

    # Row 3, Col 2: Ullage Volume
    fig.add_trace(
        go.Scatter(x=t, y=V_ullage_array, name='Ullage Volume',
                   line=dict(color='purple', width=2),
                   mode='lines',
                   fill='tozeroy',
                   fillcolor='rgba(128, 0, 128, 0.1)'),
        row=3, col=2
    )

    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=2)

    fig.update_yaxes(title_text="Pressure (bar)", row=1, col=1)
    fig.update_yaxes(title_text="Mass (kg)", row=1, col=2)
    fig.update_yaxes(title_text="Pressure (bar)", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2)
    fig.update_yaxes(title_text="Mass (g)", row=3, col=1)
    fig.update_yaxes(title_text="Volume (L)", row=3, col=2)

    # Update layout
    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text=f"N2O Tank Depletion Analysis (Burn Duration: {t[-1]:.1f} s)",
        title_font_size=16,
        hovermode='x unified'
    )

    return fig


def plot_results(n2o_sol, ethanol_sol):
    """Create comprehensive plots of simulation results"""

    if not PLOTLY_AVAILABLE:
        print("Plotly not available - skipping plotting")
        return None

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Tank Pressure', 'Liquid Mass',
            'Temperature', 'Pressurant Mass',
            'Mass Flow Rate', 'Fill Fraction'
        ),
        vertical_spacing=0.12
    )

    # N2O data
    t_n2o = n2o_sol.t

    # Calculate pressures (need to recompute from state)
    # For now, use placeholder
    pressures_n2o = np.ones_like(t_n2o) * 100  # bar

    # Plot N2O
    fig.add_trace(
        go.Scatter(x=t_n2o, y=pressures_n2o,
                   name='N2O', line=dict(color='blue')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=t_n2o, y=n2o_sol.y[0], name='N2O Liquid',
                   line=dict(color='blue')),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=t_n2o, y=n2o_sol.y[2], name='N2O Temp',
                   line=dict(color='blue')),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=t_n2o, y=n2o_sol.y[1]*1000, name='N2O Pressurant',
                   line=dict(color='blue')),
        row=2, col=2
    )

    # Ethanol data
    t_eth = ethanol_sol.t
    pressures_eth = np.ones_like(t_eth) * 100

    fig.add_trace(
        go.Scatter(x=t_eth, y=pressures_eth,
                   name='Ethanol', line=dict(color='red')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=t_eth, y=ethanol_sol.y[0], name='Ethanol Liquid',
                   line=dict(color='red')),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=t_eth, y=ethanol_sol.y[2], name='Ethanol Temp',
                   line=dict(color='red')),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=t_eth, y=ethanol_sol.y[1]*1000, name='Ethanol Pressurant',
                   line=dict(color='red')),
        row=2, col=2
    )

    # Update axes
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Pressure (bar)", row=1, col=1)
    fig.update_yaxes(title_text="Mass (kg)", row=1, col=2)
    fig.update_yaxes(title_text="Temperature (K)", row=2, col=1)
    fig.update_yaxes(title_text="Pressurant (g)", row=2, col=2)

    fig.update_layout(height=900, showlegend=True,
                     title_text="Tank Depletion Simulation Results (REVISED)")

    return fig


if __name__ == "__main__":
    # Example usage with realistic parameters

    # N2O Tank
    n2o_tank_config = TankConfig(
        volume=0.05,  # 50 L
        initial_liquid_mass=36.0,  # kg (about 90% fill)
        initial_ullage_pressure=100e5,  # 100 bar
        initial_temperature=293.15,  # 20°C
        wall_material_properties={},
        ambient_temperature=293.15,
        heat_transfer_coefficient=10.0  # W/m^2/K
    )

    n2o_pressurant_config = PressurantConfig(
        fluid_name='Nitrogen',
        supply_pressure=100e5,  # 100 bar regulated
        supply_temperature=293.15,
        regulator_flow_coefficient=0.0001  # kg/s/Pa (reduced for stability)
    )

    n2o_propellant_config = PropellantConfig(
        fluid_name='NitrousOxide',
        mass_flow_rate=0.6,  # kg/s
        is_self_pressurizing=True
    )

    # Ethanol Tank
    ethanol_tank_config = TankConfig(
        volume=0.04,  # 40 L
        initial_liquid_mass=30.0,  # kg
        initial_ullage_pressure=100e5,
        initial_temperature=293.15,
        wall_material_properties={},
        ambient_temperature=293.15,
        heat_transfer_coefficient=10.0
    )

    ethanol_pressurant_config = PressurantConfig(
        fluid_name='Nitrogen',
        supply_pressure=100e5,
        supply_temperature=293.15,
        regulator_flow_coefficient=0.0001
    )

    ethanol_propellant_config = PropellantConfig(
        fluid_name='Ethanol',
        mass_flow_rate=0.4,  # kg/s
        is_self_pressurizing=False
    )

    # Create and run simulations
    print("\n" + "="*60)
    print("N2O TANK SIMULATION")
    print("="*60)
    n2o_tank = TwoPhaseNitrousTank(n2o_tank_config, n2o_pressurant_config,
                                    n2o_propellant_config)
    n2o_sol = n2o_tank.simulate((0, 100))

    print(f"\nN2O tank duration: {n2o_sol.t[-1]:.1f} s")
    print(f"Temperature: {n2o_sol.y[2,0]:.2f} K → {n2o_sol.y[2,-1]:.2f} K " +
          f"({n2o_sol.y[2,-1]-n2o_sol.y[2,0]:+.2f} K)")

    print("\n" + "="*60)
    print("ETHANOL TANK SIMULATION")
    print("="*60)
    ethanol_tank = EthanolTank(ethanol_tank_config, ethanol_pressurant_config,
                               ethanol_propellant_config)
    ethanol_sol = ethanol_tank.simulate((0, 100))

    print(f"\nEthanol tank duration: {ethanol_sol.t[-1]:.1f} s")
    print(f"Temperature: {ethanol_sol.y[2,0]:.2f} K → {ethanol_sol.y[2,-1]:.2f} K " +
          f"({ethanol_sol.y[2,-1]-ethanol_sol.y[2,0]:+.2f} K)")

    # Plot detailed N2O results
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    if PLOTLY_AVAILABLE:
        print("\nGenerating detailed N2O tank analysis plot (Plotly)...")
        fig_n2o = plot_n2o_tank_detailed(n2o_sol, n2o_tank)
        if fig_n2o:
            fig_n2o.write_html('/mnt/user-data/outputs/n2o_tank_analysis.html')
            print("✓ Saved: n2o_tank_analysis.html")

        print("\nGenerating comparison plot (Plotly)...")
        fig_comparison = plot_results(n2o_sol, ethanol_sol)
        if fig_comparison:
            fig_comparison.write_html('/mnt/user-data/outputs/tank_comparison.html')
            print("✓ Saved: tank_comparison.html")

    elif MATPLOTLIB_AVAILABLE:
        print("\nGenerating detailed N2O tank analysis plot (Matplotlib)...")
        fig_n2o = plot_n2o_tank_matplotlib(n2o_sol, n2o_tank,
                                           '/mnt/user-data/outputs/n2o_tank_analysis.png')
    else:
        print("No plotting library available - skipping plots")