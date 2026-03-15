"""
Feed system sizing and pressure budget solver.

Orchestrates feed system hydraulic analysis by combining pure physics
functions from resa.physics.feed_system with engine design results.
Supports pressure-fed and pump-fed configurations, including gas-generator
and expander cycle power balance.

Example:
    from resa.core.module_configs import FeedSystemConfig
    from resa.solvers.feed_system import FeedSystemSolver

    solver = FeedSystemSolver()
    config = FeedSystemConfig(feed_type="pump-fed", cycle_type="gas-generator")
    result = solver.solve(config, engine_result)
"""

import logging
from typing import Any, Dict

from resa.core.exceptions import ConfigurationError
from resa.core.interfaces import Solver
from resa.core.module_configs import FeedSystemConfig
from resa.core.results import FeedSystemResult
from resa.physics.feed_system import (
    expander_cycle_power_balance,
    gas_generator_power_balance,
    line_pressure_drop,
    npsh_available,
    pressure_budget,
    pump_power,
)

logger = logging.getLogger(__name__)

# Default fluid properties for quick estimates
_FLUID_DEFAULTS: Dict[str, Dict[str, float]] = {
    "ox": {"rho": 800.0, "viscosity": 1e-4},  # N2O approximate
    "fuel": {"rho": 800.0, "viscosity": 1e-4},  # Ethanol approximate
}

# GG exhaust approximate Cp [J/(kg.K)]
_GG_CP_DEFAULT = 1800.0

# Atmospheric pressure for turbine exhaust [Pa]
_P_ATM_PA = 101325.0


class FeedSystemSolver(Solver[FeedSystemResult]):
    """Feed system sizing and pressure budget analysis.

    Calculates line losses, injector drop, cooling drop, pump power,
    NPSH, and turbopump power balance for various engine cycle types.
    """

    @property
    def name(self) -> str:
        return "Feed System Solver"

    def validate_inputs(self, config: FeedSystemConfig, engine_result: Any) -> bool:
        """Validate feed system config and engine result compatibility.

        Args:
            config: Feed system configuration.
            engine_result: EngineDesignResult with mass flows and Pc.

        Returns:
            True if inputs are valid.

        Raises:
            ConfigurationError: If required data is missing or invalid.
        """
        if config.feed_type not in ("pressure-fed", "pump-fed"):
            raise ConfigurationError(
                f"Unknown feed_type '{config.feed_type}'. "
                "Must be 'pressure-fed' or 'pump-fed'."
            )

        valid_cycles = ("none", "gas-generator", "expander", "staged-combustion")
        if config.cycle_type not in valid_cycles:
            raise ConfigurationError(
                f"Unknown cycle_type '{config.cycle_type}'. "
                f"Must be one of {valid_cycles}."
            )

        if config.feed_type == "pump-fed" and config.cycle_type == "none":
            raise ConfigurationError(
                "Pump-fed systems require a cycle_type "
                "('gas-generator', 'expander', or 'staged-combustion')."
            )

        if not hasattr(engine_result, "massflow_ox") or engine_result.massflow_ox <= 0:
            raise ConfigurationError(
                "Engine result must have positive massflow_ox."
            )

        if not hasattr(engine_result, "massflow_fuel") or engine_result.massflow_fuel <= 0:
            raise ConfigurationError(
                "Engine result must have positive massflow_fuel."
            )

        return True

    def solve(
        self, config: FeedSystemConfig, engine_result: Any
    ) -> FeedSystemResult:
        """Run feed system analysis.

        Args:
            config: FeedSystemConfig with line geometry, pump params, etc.
            engine_result: EngineDesignResult providing mass flows, Pc,
                and optional cooling pressure drop.

        Returns:
            FeedSystemResult with pressure budget, pump sizing, and
            cycle power balance.
        """
        self.validate_inputs(config, engine_result)

        mdot_ox = engine_result.massflow_ox
        mdot_fuel = engine_result.massflow_fuel
        pc_bar = engine_result.pc_bar

        # --- Step 1: Injector pressure drop (20% of Pc if not specified) ---
        injector_dp_bar = 0.20 * pc_bar
        logger.info(
            "Injector dp estimated at 20%% of Pc: %.2f bar", injector_dp_bar
        )

        # --- Step 2: Cooling pressure drop from engine result ---
        cooling_dp_bar = 0.0
        if engine_result.cooling is not None:
            cooling_dp_bar = engine_result.cooling.pressure_drop
            logger.info("Cooling dp from engine result: %.2f bar", cooling_dp_bar)
        else:
            logger.info("No cooling result available; cooling dp = 0")

        # --- Step 3: Line losses ---
        rho_ox = _FLUID_DEFAULTS["ox"]["rho"]
        mu_ox = _FLUID_DEFAULTS["ox"]["viscosity"]
        rho_fuel = _FLUID_DEFAULTS["fuel"]["rho"]
        mu_fuel = _FLUID_DEFAULTS["fuel"]["viscosity"]

        dp_line_ox_pa = line_pressure_drop(
            mdot=mdot_ox,
            rho=rho_ox,
            viscosity=mu_ox,
            length=config.ox_line_length_m,
            diameter=config.ox_line_diameter_m,
            roughness=config.ox_line_roughness_m,
            k_fittings=config.ox_k_fittings,
        )

        dp_line_fuel_pa = line_pressure_drop(
            mdot=mdot_fuel,
            rho=rho_fuel,
            viscosity=mu_fuel,
            length=config.fuel_line_length_m,
            diameter=config.fuel_line_diameter_m,
            roughness=config.fuel_line_roughness_m,
            k_fittings=config.fuel_k_fittings,
        )

        line_losses_ox_bar = dp_line_ox_pa / 1e5
        line_losses_fuel_bar = dp_line_fuel_pa / 1e5
        max_line_loss_bar = max(line_losses_ox_bar, line_losses_fuel_bar)

        logger.info(
            "Line losses: ox=%.2f bar, fuel=%.2f bar",
            line_losses_ox_bar,
            line_losses_fuel_bar,
        )

        # --- Step 4: Pressure budget ---
        budget = pressure_budget(
            pc_bar=pc_bar,
            injector_dp_bar=injector_dp_bar,
            cooling_dp_bar=cooling_dp_bar,
            line_losses_bar=max_line_loss_bar,
        )
        feed_pressure_bar = budget["feed_pressure_bar"]

        # --- Step 5: Feed-type specific calculations ---
        pump_power_ox_w = 0.0
        pump_power_fuel_w = 0.0
        pump_head_ox_m = 0.0
        pump_head_fuel_m = 0.0
        npsh_avail_m = 0.0
        npsh_margin_m = 0.0
        tank_pressure_bar = 0.0
        pump_discharge_bar = 0.0
        turbine_power_w = 0.0
        power_balance_margin = 0.0

        if config.feed_type == "pressure-fed":
            # Tank provides all feed pressure
            if config.tank_pressure_bar > 0:
                tank_pressure_bar = config.tank_pressure_bar
            else:
                tank_pressure_bar = feed_pressure_bar
            pump_discharge_bar = 0.0
            logger.info(
                "Pressure-fed: tank pressure = %.1f bar", tank_pressure_bar
            )

        elif config.feed_type == "pump-fed":
            # Tank provides ullage + suction head; pump provides the rest
            tank_pressure_bar = config.ullage_pressure_bar
            dp_pump_total_bar = feed_pressure_bar - tank_pressure_bar
            if dp_pump_total_bar < 0:
                dp_pump_total_bar = 0.0
            pump_discharge_bar = feed_pressure_bar

            dp_pump_total_pa = dp_pump_total_bar * 1e5

            # Ox pump
            ox_pump = pump_power(
                mdot_kg_s=mdot_ox,
                dp_pa=dp_pump_total_pa,
                rho_kg_m3=rho_ox,
                eta_pump=config.pump_efficiency,
            )
            pump_power_ox_w = ox_pump["power_w"]
            pump_head_ox_m = ox_pump["head_m"]

            # Fuel pump
            fuel_pump = pump_power(
                mdot_kg_s=mdot_fuel,
                dp_pa=dp_pump_total_pa,
                rho_kg_m3=rho_fuel,
                eta_pump=config.pump_efficiency,
            )
            pump_power_fuel_w = fuel_pump["power_w"]
            pump_head_fuel_m = fuel_pump["head_m"]

            # NPSH (using ox side as critical, assume vapor pressure ~ 30 bar
            # for N2O at ~280K)
            p_tank_pa = tank_pressure_bar * 1e5
            p_vapor_pa = 30.0 * 1e5  # N2O approximate vapor pressure
            npsh_avail_m = npsh_available(
                p_tank_pa=p_tank_pa,
                p_vapor_pa=p_vapor_pa,
                rho_kg_m3=rho_ox,
                h_suction_m=config.suction_head_m,
                line_dp_pa=dp_line_ox_pa,
            )
            # NPSH margin: required NPSH ~ 2m typical for small pumps
            npsh_required_m = 2.0
            npsh_margin_m = npsh_avail_m - npsh_required_m

            total_pump_power_w = pump_power_ox_w + pump_power_fuel_w
            mdot_total = mdot_ox + mdot_fuel
            rho_avg = (rho_ox + rho_fuel) / 2.0

            # --- Step 6: Cycle power balance ---
            if config.cycle_type == "gas-generator":
                # GG flow ~ 2-5% of total; use 3% estimate
                mdot_gg = 0.03 * mdot_total
                p_turb_in = pc_bar * 0.9 * 1e5  # GG pressure ~ 90% Pc
                p_turb_out = _P_ATM_PA

                gg_balance = gas_generator_power_balance(
                    mdot_gg=mdot_gg,
                    cp_gg=_GG_CP_DEFAULT,
                    T_gg=config.gg_temperature_k,
                    p_in=p_turb_in,
                    p_out=p_turb_out,
                    eta_turbine=config.turbine_efficiency,
                    eta_pump=config.pump_efficiency,
                    mdot_pump=mdot_total,
                    dp_pump=dp_pump_total_pa,
                    rho_pump=rho_avg,
                )
                turbine_power_w = gg_balance["turbine_power_w"]
                power_balance_margin = gg_balance["margin_pct"] / 100.0

            elif config.cycle_type == "expander":
                # Enthalpy rise from cooling jacket
                dh_coolant = 0.0
                if engine_result.cooling is not None:
                    t_in = engine_result.cooling.T_coolant[0]
                    t_out = engine_result.cooling.outlet_temp
                    # Approximate dh = cp * dT, cp ~ 2500 J/(kg.K) for ethanol
                    cp_coolant = 2500.0
                    dh_coolant = cp_coolant * abs(t_out - t_in)

                if dh_coolant <= 0:
                    dh_coolant = 200_000.0  # 200 kJ/kg fallback
                    logger.warning(
                        "No cooling data for expander cycle; "
                        "using fallback dh=%.0f J/kg",
                        dh_coolant,
                    )

                p_turb_in = feed_pressure_bar * 1e5
                p_turb_out = pc_bar * 1e5  # Turbine exhausts into chamber

                exp_balance = expander_cycle_power_balance(
                    mdot_coolant=mdot_fuel,
                    dh_coolant_j_kg=dh_coolant,
                    p_in=p_turb_in,
                    p_out=p_turb_out,
                    eta_turbine=config.turbine_efficiency,
                    eta_pump=config.pump_efficiency,
                    mdot_pump=mdot_total,
                    dp_pump=dp_pump_total_pa,
                    rho_pump=rho_avg,
                )
                turbine_power_w = exp_balance["turbine_power_w"]
                margin_w = exp_balance["margin_w"]
                power_balance_margin = (
                    margin_w / total_pump_power_w if total_pump_power_w > 0 else 0.0
                )

            elif config.cycle_type == "staged-combustion":
                logger.warning(
                    "Staged-combustion cycle not yet implemented; "
                    "returning pump sizing only."
                )

        # --- Build result ---
        result = FeedSystemResult(
            tank_pressure_bar=tank_pressure_bar,
            pump_discharge_pressure_bar=pump_discharge_bar,
            injector_dp_bar=injector_dp_bar,
            cooling_dp_bar=cooling_dp_bar,
            line_losses_ox_bar=line_losses_ox_bar,
            line_losses_fuel_bar=line_losses_fuel_bar,
            pump_power_ox_w=pump_power_ox_w,
            pump_power_fuel_w=pump_power_fuel_w,
            pump_head_ox_m=pump_head_ox_m,
            pump_head_fuel_m=pump_head_fuel_m,
            npsh_available_m=npsh_avail_m,
            npsh_margin_m=npsh_margin_m,
            feed_type=config.feed_type,
            cycle_type=config.cycle_type,
            turbine_power_w=turbine_power_w,
            power_balance_margin=power_balance_margin,
        )

        logger.info("Feed system analysis complete: %s", config.feed_type)
        return result
