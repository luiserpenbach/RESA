"""
Tank simulation API routes.
"""
from __future__ import annotations

import asyncio
import logging
from functools import partial

import numpy as np
from fastapi import APIRouter, HTTPException

from api.models.tank_models import (
    TankSimConfigRequest,
    TankSimResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tank", tags=["tank"])

MAX_OUTPUT_POINTS = 200


def _downsample(arr: np.ndarray, n: int) -> list[float]:
    """Evenly downsample array to at most n points."""
    if len(arr) <= n:
        return arr.tolist()
    idx = np.linspace(0, len(arr) - 1, n, dtype=int)
    return arr[idx].tolist()


def _run_simulation(req: TankSimConfigRequest):
    from resa.addons.tank.config import (
        TankConfig,
        PressurantConfig,
        PropellantConfig,
    )
    from resa.addons.tank.simulator import TwoPhaseNitrousTank, EthanolTank

    wall_props = {
        "density": 2700.0,
        "specific_heat": 900.0,
        "thermal_conductivity": 200.0,
    }

    tank_cfg = TankConfig(
        volume=req.tank.volume,
        initial_liquid_mass=req.tank.initial_liquid_mass,
        initial_ullage_pressure=req.tank.initial_ullage_pressure,
        initial_temperature=req.tank.initial_temperature,
        wall_material_properties=wall_props,
        ambient_temperature=req.tank.ambient_temperature,
        heat_transfer_coefficient=req.tank.heat_transfer_coefficient,
    )
    pres_cfg = PressurantConfig(
        fluid_name=req.pressurant.fluid_name,
        supply_pressure=req.pressurant.supply_pressure,
        supply_temperature=req.pressurant.supply_temperature,
        regulator_flow_coefficient=req.pressurant.regulator_flow_coefficient,
    )
    prop_cfg = PropellantConfig(
        fluid_name=req.propellant.fluid_name,
        mass_flow_rate=req.propellant.mass_flow_rate,
        is_self_pressurizing=req.propellant.is_self_pressurizing,
    )

    tank_type = req.tank_type.lower()
    if tank_type == "n2o":
        sim = TwoPhaseNitrousTank(tank_cfg, pres_cfg, prop_cfg)
    elif tank_type == "ethanol":
        sim = EthanolTank(tank_cfg, pres_cfg, prop_cfg)
    else:
        raise ValueError(f"Unknown tank_type: {req.tank_type!r}. Use 'n2o' or 'ethanol'.")

    sol = sim.simulate(t_span=(0.0, req.duration_s), dt=0.1)
    return sol, tank_type


@router.post("/simulate", response_model=TankSimResponse)
async def simulate_tank(req: TankSimConfigRequest | None = None):
    """Simulate propellant tank depletion over a burn."""
    req = req or TankSimConfigRequest()
    loop = asyncio.get_event_loop()
    try:
        sol, tank_type = await loop.run_in_executor(None, partial(_run_simulation, req))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Tank simulation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    t = sol.t
    y = sol.y  # shape: (n_state, n_times)

    burn_duration = float(t[-1])
    n = MAX_OUTPUT_POINTS

    if tank_type == "n2o":
        # State: [m_liquid, m_pressurant, T_liquid, T_ullage]
        m_liquid = y[0]
        T_liquid = y[2]
        T_ullage = y[3]

        # Recompute pressure from state (approximate via sat pressure + ideal gas)
        # Use a simplified re-calculation for the response
        import CoolProp.CoolProp as CP
        pressures = []
        for i in range(len(t)):
            try:
                P_sat = CP.PropsSI("P", "T", float(T_liquid[i]), "Q", 0, "NitrousOxide")
            except Exception:
                P_sat = req.tank.initial_ullage_pressure * 0.8
            pressures.append(P_sat / 1e5)
        pressures = np.array(pressures)
    else:
        # EthanolTank state: [m_liquid, m_pressurant, T_liquid]
        m_liquid = y[0]
        T_liquid = y[2]
        T_ullage = y[2]  # single phase - same temp

        # Re-derive pressure from pressurant ideal gas
        R_N2 = 296.8  # J/kg/K for nitrogen
        pressures = np.zeros(len(t))
        for i in range(len(t)):
            rho_liq = 789.0  # approximate ethanol density kg/m3
            V_liq = float(m_liquid[i]) / rho_liq
            V_ullage = req.tank.volume - V_liq
            if V_ullage > 1e-6 and y[1][i] > 0:
                pressures[i] = float(y[1][i]) * R_N2 * float(T_liquid[i]) / V_ullage / 1e5
            else:
                pressures[i] = req.tank.initial_ullage_pressure / 1e5

    return TankSimResponse(
        time_s=_downsample(t, n),
        pressure_bar=_downsample(pressures, n),
        liquid_mass_kg=_downsample(m_liquid, n),
        liquid_temperature_k=_downsample(T_liquid, n),
        ullage_temperature_k=_downsample(T_ullage, n),
        burn_duration_s=burn_duration,
        final_liquid_mass_kg=float(m_liquid[-1]),
        final_pressure_bar=float(pressures[-1]),
    )
