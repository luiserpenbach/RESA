"""
Wall thickness solver for RESA.

Calculates minimum wall thickness and stress distribution along the
nozzle contour considering pressure and thermal loads.
"""
import logging

import numpy as np

from resa.core.interfaces import Solver
from resa.core.materials import get_material
from resa.core.results import CoolingResult, NozzleGeometry, WallThicknessResult
from resa.physics.structural import (
    combined_wall_stress,
    min_wall_thickness_pressure,
    min_wall_thickness_thermal,
)

logger = logging.getLogger(__name__)


class WallThicknessSolver(Solver):
    """Calculates wall thickness requirements along nozzle contour.

    Considers:
    - Pressure loads (hoop stress via Barlow's formula)
    - Thermal stress from heat flux through wall
    - Combined von Mises stress
    """

    @property
    def name(self) -> str:
        return "Wall Thickness Solver"

    def validate_inputs(
        self,
        nozzle_geometry: NozzleGeometry,
        cooling_result: CoolingResult,
        pc_bar: float,
        **kwargs,
    ) -> bool:
        if nozzle_geometry is None or cooling_result is None:
            return False
        if pc_bar <= 0:
            return False
        return True

    def solve(
        self,
        nozzle_geometry: NozzleGeometry,
        cooling_result: CoolingResult,
        pc_bar: float,
        material_name: str = "inconel718",
        safety_factor_pressure: float = 2.0,
        safety_factor_thermal: float = 1.5,
        actual_wall_thickness: float = 0.001,
    ) -> WallThicknessResult:
        """Run wall thickness analysis along the nozzle.

        Args:
            nozzle_geometry: Nozzle contour geometry
            cooling_result: Thermal analysis results (heat flux, wall temps)
            pc_bar: Chamber pressure [bar]
            material_name: Material identifier from materials database
            safety_factor_pressure: Safety factor for pressure loads
            safety_factor_thermal: Safety factor for thermal loads
            actual_wall_thickness: Actual wall thickness [m]

        Returns:
            WallThicknessResult with stress and thickness arrays
        """
        material = get_material(material_name)
        pc_pa = pc_bar * 1e5

        x = nozzle_geometry.x_full
        y = nozzle_geometry.y_full  # local radius [m]
        n = len(x)

        # Get heat flux and temperature arrays from cooling result
        q_flux = cooling_result.q_flux
        # Ensure arrays match length (interpolate if needed)
        if len(q_flux) != n:
            q_flux = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(q_flux)),
                q_flux,
            )

        # Initialize output arrays
        min_t_pressure = np.zeros(n)
        min_t_thermal = np.zeros(n)
        sigma_hoop = np.zeros(n)
        sigma_thermal = np.zeros(n)
        sigma_vm = np.zeros(n)

        for i in range(n):
            # Local pressure estimate (approximate: use Pc everywhere,
            # could be refined with local static pressure from Mach)
            p_local = pc_pa

            # Minimum thickness for pressure
            min_t_pressure[i] = min_wall_thickness_pressure(
                p_local, y[i], material.yield_strength_pa, safety_factor_pressure
            )

            # Maximum thickness from thermal constraint
            min_t_thermal[i] = min_wall_thickness_thermal(
                abs(q_flux[i]),
                material.thermal_conductivity_w_mk,
                material.cte_1_k,
                material.elastic_modulus_pa,
                material.yield_strength_pa,
                safety_factor_thermal,
                material.poisson_ratio,
            )

            # Combined stress at actual wall thickness
            stress = combined_wall_stress(
                p_local,
                y[i],
                actual_wall_thickness,
                abs(q_flux[i]),
                material.thermal_conductivity_w_mk,
                material.cte_1_k,
                material.elastic_modulus_pa,
                material.poisson_ratio,
            )
            sigma_hoop[i] = stress["hoop_stress"]
            sigma_thermal[i] = stress["thermal_stress"]
            sigma_vm[i] = stress["von_mises"]

        # Combined minimum thickness (envelope)
        min_t_combined = np.maximum(min_t_pressure, 0.0)
        # For thermal: min_t_thermal is the MAX allowed thickness,
        # so it constrains from above rather than below.
        # The design thickness must be >= min_t_pressure AND <= min_t_thermal

        # Safety factor: von Mises
        actual_t = np.full(n, actual_wall_thickness)
        safety_factor = np.where(
            sigma_vm > 0,
            material.yield_strength_pa / sigma_vm,
            np.inf,
        )

        logger.info(
            "Wall thickness analysis complete: min SF=%.2f at x=%.3f m",
            np.min(safety_factor),
            x[np.argmin(safety_factor)],
        )

        return WallThicknessResult(
            x=x,
            min_thickness_pressure=min_t_pressure,
            min_thickness_thermal=min_t_thermal,
            min_thickness_combined=min_t_combined,
            actual_thickness=actual_t,
            safety_factor=safety_factor,
            hoop_stress=sigma_hoop,
            thermal_stress=sigma_thermal,
            von_mises_stress=sigma_vm,
            material=material,
        )
