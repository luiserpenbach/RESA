"""
Feed system hydraulics and cycle thermodynamics.

Pure functions for calculating pressure drops, pump sizing, NPSH,
and power balance for gas-generator and expander cycle engines.
All functions are stateless with no side effects.

Functions:
    line_pressure_drop: Darcy-Weisbach pipe friction loss
    pressure_budget: Feed system pressure budget rollup
    pump_power: Centrifugal pump power and head
    npsh_available: Net positive suction head available
    gas_generator_power_balance: GG cycle turbine-pump matching
    expander_cycle_power_balance: Expander cycle turbine-pump matching
"""

import logging
import math

logger = logging.getLogger(__name__)

G0 = 9.80665  # Standard gravitational acceleration [m/s^2]


def line_pressure_drop(
    mdot: float,
    rho: float,
    viscosity: float,
    length: float,
    diameter: float,
    roughness: float,
    k_fittings: float = 0.0,
) -> float:
    """Calculate pressure drop in a pipe using Darcy-Weisbach with Swamee-Jain friction.

    Uses the Swamee-Jain explicit approximation for the Colebrook-White
    friction factor, valid for 5000 < Re < 1e8 and 1e-6 < e/D < 0.05.
    For laminar flow (Re < 2300), uses f = 64/Re.

    Args:
        mdot: Mass flow rate [kg/s].
        rho: Fluid density [kg/m^3].
        viscosity: Dynamic viscosity [Pa.s].
        length: Pipe length [m].
        diameter: Pipe inner diameter [m].
        roughness: Pipe wall roughness [m].
        k_fittings: Sum of minor loss coefficients (K-factors).

    Returns:
        Pressure drop [Pa].

    Raises:
        ValueError: If diameter or density is non-positive.
    """
    if diameter <= 0:
        raise ValueError(f"Diameter must be positive, got {diameter}")
    if rho <= 0:
        raise ValueError(f"Density must be positive, got {rho}")

    area = math.pi / 4.0 * diameter**2
    velocity = mdot / (rho * area)

    if velocity == 0.0:
        return 0.0

    re = rho * velocity * diameter / viscosity

    if re < 2300:
        # Laminar flow
        f = 64.0 / re
        logger.debug("Laminar flow regime: Re=%.0f, f=%.6f", re, f)
    else:
        # Swamee-Jain approximation for turbulent flow
        e_over_d = roughness / diameter
        log_arg = e_over_d / 3.7 + 5.74 / re**0.9
        f = 0.25 / (math.log10(log_arg)) ** 2
        logger.debug("Turbulent flow regime: Re=%.0f, f=%.6f", re, f)

    dynamic_pressure = 0.5 * rho * velocity**2
    dp_friction = f * (length / diameter) * dynamic_pressure
    dp_minor = k_fittings * dynamic_pressure

    dp_total = dp_friction + dp_minor
    logger.debug(
        "Line dp: friction=%.0f Pa, minor=%.0f Pa, total=%.0f Pa",
        dp_friction,
        dp_minor,
        dp_total,
    )

    return dp_total


def pressure_budget(
    pc_bar: float,
    injector_dp_bar: float,
    cooling_dp_bar: float,
    line_losses_bar: float,
    margin_pct: float = 0.10,
) -> dict:
    """Calculate required feed pressure with safety margin.

    Args:
        pc_bar: Chamber pressure [bar].
        injector_dp_bar: Injector pressure drop [bar].
        cooling_dp_bar: Cooling jacket pressure drop [bar].
        line_losses_bar: Feed line friction losses [bar].
        margin_pct: Fractional margin on total feed pressure (default 10%).

    Returns:
        Dict with keys:
            feed_pressure_bar: Required feed pressure including margin.
            total_dp_bar: Sum of all pressure drops (no margin).
            margin_bar: Absolute margin [bar].
    """
    total_dp = injector_dp_bar + cooling_dp_bar + line_losses_bar
    feed_no_margin = pc_bar + total_dp
    margin_bar = feed_no_margin * margin_pct
    feed_pressure = feed_no_margin + margin_bar

    logger.info(
        "Pressure budget: Pc=%.1f + dp=%.1f + margin=%.1f = %.1f bar",
        pc_bar,
        total_dp,
        margin_bar,
        feed_pressure,
    )

    return {
        "feed_pressure_bar": feed_pressure,
        "total_dp_bar": total_dp,
        "margin_bar": margin_bar,
    }


def pump_power(
    mdot_kg_s: float,
    dp_pa: float,
    rho_kg_m3: float,
    eta_pump: float = 0.65,
) -> dict:
    """Calculate centrifugal pump power requirement and head.

    Args:
        mdot_kg_s: Mass flow rate [kg/s].
        dp_pa: Required pressure rise [Pa].
        rho_kg_m3: Fluid density [kg/m^3].
        eta_pump: Pump isentropic efficiency (0-1).

    Returns:
        Dict with keys:
            power_w: Shaft power required [W].
            head_m: Pump head [m].
    """
    if eta_pump <= 0:
        raise ValueError(f"Pump efficiency must be positive, got {eta_pump}")

    power_w = mdot_kg_s * dp_pa / (rho_kg_m3 * eta_pump)
    head_m = dp_pa / (rho_kg_m3 * G0)

    logger.debug(
        "Pump sizing: mdot=%.3f kg/s, dp=%.0f Pa, P=%.0f W, H=%.1f m",
        mdot_kg_s,
        dp_pa,
        power_w,
        head_m,
    )

    return {"power_w": power_w, "head_m": head_m}


def npsh_available(
    p_tank_pa: float,
    p_vapor_pa: float,
    rho_kg_m3: float,
    h_suction_m: float,
    line_dp_pa: float,
) -> float:
    """Calculate net positive suction head available.

    NPSHa = (p_tank - p_vapor) / (rho * g) + h_suction - line_dp / (rho * g)

    Args:
        p_tank_pa: Tank ullage pressure [Pa].
        p_vapor_pa: Fluid vapor pressure at suction temperature [Pa].
        rho_kg_m3: Fluid density [kg/m^3].
        h_suction_m: Elevation of tank above pump inlet [m].
        line_dp_pa: Suction line pressure loss [Pa].

    Returns:
        NPSHa [m].
    """
    pressure_head = (p_tank_pa - p_vapor_pa) / (rho_kg_m3 * G0)
    loss_head = line_dp_pa / (rho_kg_m3 * G0)
    npsha = pressure_head + h_suction_m - loss_head

    logger.debug(
        "NPSH available: pressure_head=%.1f m, suction=%.1f m, "
        "loss=%.1f m, NPSHa=%.1f m",
        pressure_head,
        h_suction_m,
        loss_head,
        npsha,
    )

    return npsha


def gas_generator_power_balance(
    mdot_gg: float,
    cp_gg: float,
    T_gg: float,
    p_in: float,
    p_out: float,
    eta_turbine: float,
    eta_pump: float,
    mdot_pump: float,
    dp_pump: float,
    rho_pump: float,
) -> dict:
    """Gas generator cycle power balance.

    Calculates turbine power from GG exhaust expansion and compares
    against pump power requirement. Uses gamma=1.2 approximation for
    fuel-rich GG combustion products.

    Args:
        mdot_gg: Gas generator mass flow rate [kg/s].
        cp_gg: GG exhaust specific heat at constant pressure [J/(kg.K)].
        T_gg: GG combustion temperature [K].
        p_in: Turbine inlet pressure [Pa].
        p_out: Turbine exhaust pressure [Pa].
        eta_turbine: Turbine isentropic efficiency (0-1).
        eta_pump: Pump isentropic efficiency (0-1).
        mdot_pump: Total pump mass flow rate [kg/s].
        dp_pump: Total pump pressure rise [Pa].
        rho_pump: Average fluid density through pump [kg/m^3].

    Returns:
        Dict with keys:
            turbine_power_w: Available turbine power [W].
            pump_power_w: Required pump power [W].
            margin_w: Power surplus (turbine - pump) [W].
            margin_pct: Power margin as percentage of pump power.
    """
    gamma_gg = 1.2
    pressure_ratio = p_out / p_in

    # Isentropic turbine work with efficiency
    exponent = (gamma_gg - 1.0) / gamma_gg
    turbine_power_w = (
        mdot_gg * cp_gg * T_gg * eta_turbine * (1.0 - pressure_ratio**exponent)
    )

    pump_power_w = mdot_pump * dp_pump / (rho_pump * eta_pump)
    margin_w = turbine_power_w - pump_power_w
    margin_pct = (margin_w / pump_power_w * 100.0) if pump_power_w > 0 else 0.0

    logger.info(
        "GG power balance: P_turb=%.0f W, P_pump=%.0f W, margin=%.1f%%",
        turbine_power_w,
        pump_power_w,
        margin_pct,
    )

    return {
        "turbine_power_w": turbine_power_w,
        "pump_power_w": pump_power_w,
        "margin_w": margin_w,
        "margin_pct": margin_pct,
    }


def expander_cycle_power_balance(
    mdot_coolant: float,
    dh_coolant_j_kg: float,
    p_in: float,
    p_out: float,
    eta_turbine: float,
    eta_pump: float,
    mdot_pump: float,
    dp_pump: float,
    rho_pump: float,
) -> dict:
    """Expander cycle power balance using coolant enthalpy rise.

    The turbine is driven by heated coolant expanding from turbine inlet
    to outlet pressure. Uses a simplified pressure-ratio correction factor.

    Args:
        mdot_coolant: Coolant mass flow rate through turbine [kg/s].
        dh_coolant_j_kg: Coolant specific enthalpy rise in cooling jacket [J/kg].
        p_in: Turbine inlet pressure [Pa].
        p_out: Turbine outlet pressure [Pa].
        eta_turbine: Turbine isentropic efficiency (0-1).
        eta_pump: Pump isentropic efficiency (0-1).
        mdot_pump: Total pump mass flow rate [kg/s].
        dp_pump: Total pump pressure rise [Pa].
        rho_pump: Average fluid density through pump [kg/m^3].

    Returns:
        Dict with keys:
            turbine_power_w: Available turbine power [W].
            pump_power_w: Required pump power [W].
            margin_w: Power surplus (turbine - pump) [W].
    """
    pressure_ratio_factor = 1.0 - (p_out / p_in) ** 0.3
    turbine_power_w = (
        mdot_coolant * dh_coolant_j_kg * eta_turbine * pressure_ratio_factor
    )

    pump_power_w = mdot_pump * dp_pump / (rho_pump * eta_pump)
    margin_w = turbine_power_w - pump_power_w

    logger.info(
        "Expander power balance: P_turb=%.0f W, P_pump=%.0f W, margin=%.0f W",
        turbine_power_w,
        pump_power_w,
        margin_w,
    )

    return {
        "turbine_power_w": turbine_power_w,
        "pump_power_w": pump_power_w,
        "margin_w": margin_w,
    }
