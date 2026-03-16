"""
Swirl injector design API routes.
"""
from __future__ import annotations

import asyncio
import logging
from functools import partial

from fastapi import APIRouter, HTTPException

from api.models.injector_models import (
    InjectorConfigRequest,
    InjectorGeometryResponse,
    InjectorMassFlowResponse,
    InjectorPerformanceResponse,
    InjectorResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/injector", tags=["injector"])


def _run_design(req: InjectorConfigRequest):
    from resa.addons.injector import (
        InjectorConfig,
        PropellantConfig,
        OperatingConditions,
        GeometryConfig,
        LCSCCalculator,
        GCSCCalculator,
    )

    config = InjectorConfig(
        propellants=PropellantConfig(
            fuel=req.propellants.fuel,
            oxidizer=req.propellants.oxidizer,
            fuel_temperature=req.propellants.fuel_temperature,
            oxidizer_temperature=req.propellants.oxidizer_temperature,
        ),
        operating=OperatingConditions(
            inlet_pressure=req.operating.inlet_pressure,
            pressure_drop=req.operating.pressure_drop,
            mass_flow_fuel=req.operating.mass_flow_fuel,
            mass_flow_oxidizer=req.operating.mass_flow_oxidizer,
            oxidizer_velocity=req.operating.oxidizer_velocity,
        ),
        geometry=GeometryConfig(
            num_elements=req.geometry.num_elements,
            num_fuel_ports=req.geometry.num_fuel_ports,
            num_ox_orifices=req.geometry.num_ox_orifices,
            post_thickness=req.geometry.post_thickness,
            spray_half_angle=req.geometry.spray_half_angle,
            minimum_clearance=req.geometry.minimum_clearance,
        ),
    )

    injector_type = req.injector_type.upper()
    if injector_type == "LCSC":
        calc = LCSCCalculator(config)
    elif injector_type == "GCSC":
        calc = GCSCCalculator(config)
    else:
        raise ValueError(f"Unknown injector type: {req.injector_type!r}. Use 'LCSC' or 'GCSC'.")

    return calc.calculate()


@router.post("/design", response_model=InjectorResponse)
async def design_injector(req: InjectorConfigRequest | None = None):
    """Size a swirl coaxial injector (LCSC or GCSC)."""
    req = req or InjectorConfigRequest()
    loop = asyncio.get_event_loop()
    try:
        r = await loop.run_in_executor(None, partial(_run_design, req))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Injector design failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    g = r.geometry
    p = r.performance
    m = r.mass_flows

    return InjectorResponse(
        injector_type=r.injector_type,
        geometry=InjectorGeometryResponse(
            fuel_orifice_radius_mm=g.fuel_orifice_radius * 1000,
            fuel_port_radius_mm=g.fuel_port_radius * 1000,
            swirl_chamber_radius_mm=g.swirl_chamber_radius * 1000,
            ox_outlet_radius_mm=g.ox_outlet_radius * 1000,
            ox_inlet_orifice_radius_mm=g.ox_inlet_orifice_radius * 1000,
            recess_length_mm=g.recess_length * 1000,
        ),
        performance=InjectorPerformanceResponse(
            spray_half_angle_deg=p.spray_half_angle,
            swirl_number=p.swirl_number,
            momentum_flux_ratio=p.momentum_flux_ratio,
            velocity_ratio=p.velocity_ratio,
            weber_number=p.weber_number,
            discharge_coefficient=p.discharge_coefficient,
        ),
        mass_flows=InjectorMassFlowResponse(
            fuel_per_element_kg_s=m.fuel_per_element,
            oxidizer_per_element_kg_s=m.oxidizer_per_element,
            total_fuel_kg_s=m.total_fuel,
            total_oxidizer_kg_s=m.total_oxidizer,
            mixture_ratio=m.mixture_ratio,
        ),
    )
