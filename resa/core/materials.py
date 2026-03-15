"""
Material property database for RESA.

Provides common chamber wall materials with mechanical and thermal properties
for structural and thermal analysis.
"""
import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaterialProperties:
    """Mechanical and thermal properties of a chamber wall material."""

    name: str
    density_kg_m3: float
    yield_strength_pa: float
    ultimate_strength_pa: float
    elastic_modulus_pa: float
    thermal_conductivity_w_mk: float
    cte_1_k: float  # coefficient of thermal expansion [1/K]
    poisson_ratio: float
    max_service_temp_k: float


# =========================================================================
# MATERIAL DATABASE
# =========================================================================

MATERIALS: Dict[str, MaterialProperties] = {
    "inconel718": MaterialProperties(
        name="Inconel 718",
        density_kg_m3=8190,
        yield_strength_pa=1035e6,
        ultimate_strength_pa=1240e6,
        elastic_modulus_pa=200e9,
        thermal_conductivity_w_mk=11.4,
        cte_1_k=13.0e-6,
        poisson_ratio=0.29,
        max_service_temp_k=990,
    ),
    "inconel625": MaterialProperties(
        name="Inconel 625",
        density_kg_m3=8440,
        yield_strength_pa=490e6,
        ultimate_strength_pa=830e6,
        elastic_modulus_pa=205e9,
        thermal_conductivity_w_mk=9.8,
        cte_1_k=12.8e-6,
        poisson_ratio=0.28,
        max_service_temp_k=1090,
    ),
    "copper_c18150": MaterialProperties(
        name="Copper C18150 (CuCrZr)",
        density_kg_m3=8890,
        yield_strength_pa=310e6,
        ultimate_strength_pa=380e6,
        elastic_modulus_pa=130e9,
        thermal_conductivity_w_mk=320.0,
        cte_1_k=17.6e-6,
        poisson_ratio=0.34,
        max_service_temp_k=700,
    ),
    "copper_ofhc": MaterialProperties(
        name="Copper OFHC (C10200)",
        density_kg_m3=8940,
        yield_strength_pa=69e6,
        ultimate_strength_pa=220e6,
        elastic_modulus_pa=117e9,
        thermal_conductivity_w_mk=391.0,
        cte_1_k=17.0e-6,
        poisson_ratio=0.34,
        max_service_temp_k=600,
    ),
    "stainless316l": MaterialProperties(
        name="Stainless Steel 316L",
        density_kg_m3=7990,
        yield_strength_pa=170e6,
        ultimate_strength_pa=485e6,
        elastic_modulus_pa=193e9,
        thermal_conductivity_w_mk=16.3,
        cte_1_k=16.0e-6,
        poisson_ratio=0.30,
        max_service_temp_k=870,
    ),
    "haynes230": MaterialProperties(
        name="Haynes 230",
        density_kg_m3=8970,
        yield_strength_pa=390e6,
        ultimate_strength_pa=860e6,
        elastic_modulus_pa=211e9,
        thermal_conductivity_w_mk=8.9,
        cte_1_k=12.7e-6,
        poisson_ratio=0.31,
        max_service_temp_k=1420,
    ),
    "aluminum6061": MaterialProperties(
        name="Aluminum 6061-T6",
        density_kg_m3=2700,
        yield_strength_pa=276e6,
        ultimate_strength_pa=310e6,
        elastic_modulus_pa=68.9e9,
        thermal_conductivity_w_mk=167.0,
        cte_1_k=23.6e-6,
        poisson_ratio=0.33,
        max_service_temp_k=420,
    ),
}


def get_material(name: str) -> MaterialProperties:
    """Get material properties by name.

    Args:
        name: Material identifier (e.g. 'inconel718', 'copper_c18150')

    Returns:
        MaterialProperties for the requested material

    Raises:
        ValueError: If material name is not found
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in MATERIALS:
        available = ", ".join(sorted(MATERIALS.keys()))
        raise ValueError(f"Unknown material '{name}'. Available: {available}")
    return MATERIALS[key]


def list_materials() -> Dict[str, str]:
    """Return dict of material_id -> display_name."""
    return {k: v.name for k, v in MATERIALS.items()}
