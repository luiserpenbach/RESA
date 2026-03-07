"""
Config import/export routes.
"""
from __future__ import annotations

import io
import logging
import tempfile

import yaml
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import Response

from api.models.engine_models import EngineConfigRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/config", tags=["config"])


@router.post("/import-yaml", response_model=EngineConfigRequest)
async def import_yaml(file: UploadFile = File(...)) -> EngineConfigRequest:
    """Upload a YAML config file and return as EngineConfigRequest JSON."""
    from resa.core.config import EngineConfig

    content = await file.read()
    try:
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        config = EngineConfig.from_yaml(tmp_path)
        return EngineConfigRequest(**config.to_dict())
    except Exception as exc:
        logger.exception("YAML import failed")
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc


@router.post("/export-yaml")
async def export_yaml(req: EngineConfigRequest) -> Response:
    """Convert an EngineConfigRequest to a downloadable YAML file."""
    from resa.core.config import EngineConfig

    try:
        config = EngineConfig(**req.model_dump())
        data = {
            "meta": {
                "engine_name": config.engine_name,
                "version": config.version,
                "designer": config.designer,
                "description": config.description,
            },
            "propulsion": {
                "fuel": config.fuel,
                "oxidizer": config.oxidizer,
                "fuel_injection_temp_k": config.fuel_injection_temp_k,
                "oxidizer_injection_temp_k": config.oxidizer_injection_temp_k,
                "thrust_n": config.thrust_n,
                "pc_bar": config.pc_bar,
                "mr": config.mr,
                "eff_combustion": config.eff_combustion,
                "eff_nozzle_divergence": config.eff_nozzle_divergence,
                "freeze_at_throat": config.freeze_at_throat,
            },
            "nozzle": {
                "nozzle_type": config.nozzle_type,
                "throat_diameter": config.throat_diameter,
                "expansion_ratio": config.expansion_ratio,
                "p_exit_bar": config.p_exit_bar,
                "L_star_mm": config.L_star,
                "contraction_ratio": config.contraction_ratio,
                "theta_convergent": config.theta_convergent,
                "theta_exit": config.theta_exit,
                "bell_fraction": config.bell_fraction,
            },
            "cooling": {
                "coolant": config.coolant_name,
                "mode": config.cooling_mode,
                "mass_fraction": config.coolant_mass_fraction,
                "inlet": {
                    "pressure_bar": config.coolant_p_in_bar,
                    "temperature_k": config.coolant_t_in_k,
                },
                "geometry": {
                    "channel_width_throat_mm": config.channel_width_throat * 1000,
                    "channel_height_mm": config.channel_height * 1000,
                    "rib_width_throat_mm": config.rib_width_throat * 1000,
                    "wall_thickness_mm": config.wall_thickness * 1000,
                    "roughness_microns": config.wall_roughness * 1e6,
                },
                "material": {
                    "name": config.wall_material,
                    "conductivity": config.wall_conductivity,
                },
            },
        }
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
        filename = f"{config.engine_name.replace(' ', '_')}_config.yaml"
        return Response(
            content=yaml_str.encode(),
            media_type="application/x-yaml",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        logger.exception("YAML export failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
