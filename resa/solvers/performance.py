"""
Performance map solver for off-design rocket engine analysis.

Generates altitude-performance curves, throttle maps, and (future)
mixture-ratio sweeps by orchestrating the pure physics functions in
``resa.physics.performance``.

Usage::

    from resa.solvers.performance import PerformanceMapSolver
    from resa.core.module_configs import PerformanceMapConfig

    solver = PerformanceMapSolver()
    result = solver.solve(engine_result, combustion, PerformanceMapConfig())
"""
import logging
import math

import numpy as np

from resa.core.interfaces import Solver
from resa.core.module_configs import PerformanceMapConfig
from resa.core.results import (
    CombustionResult,
    EngineDesignResult,
    PerformanceMapResult,
)
from resa.physics.performance import altitude_performance_curve, thrust_at_altitude

logger = logging.getLogger(__name__)

# Constants
G0 = 9.80665  # Standard gravitational acceleration [m/s^2]


class PerformanceMapSolver(Solver["PerformanceMapResult"]):
    """
    Generates off-design performance maps.

    Produces three data products:

    1. **Altitude curve** -- thrust, Isp, and Cf vs. altitude from sea level
       to the upper atmosphere.
    2. **Throttle map** -- thrust, Isp, and chamber pressure vs. throttle
       percentage, computed by linearly scaling Pc.
    3. **MR sweep** -- reserved for future integration with a combustion
       solver (currently returns ``None`` for MR-related fields).
    """

    # ------------------------------------------------------------------
    # Solver ABC
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        return "PerformanceMapSolver"

    def validate_inputs(
        self,
        engine_result: EngineDesignResult,
        combustion: CombustionResult,
        config: PerformanceMapConfig,
    ) -> bool:
        """
        Validate that the required fields are available.

        Args:
            engine_result: Completed engine design result.
            combustion: Combustion analysis result.
            config: Performance-map configuration.

        Returns:
            ``True`` when all required inputs are present and sensible.
        """
        if engine_result.dt_mm <= 0:
            logger.error("Throat diameter must be positive (got %.3f mm)", engine_result.dt_mm)
            return False
        if combustion.cstar <= 0 or combustion.gamma <= 0:
            logger.error(
                "Invalid combustion data: cstar=%.1f m/s, gamma=%.3f",
                combustion.cstar,
                combustion.gamma,
            )
            return False
        if config.altitude_points < 2 or config.throttle_points < 2:
            logger.error("Need at least 2 points for altitude and throttle sweeps")
            return False
        return True

    def solve(
        self,
        engine_result: EngineDesignResult,
        combustion: CombustionResult,
        config: PerformanceMapConfig,
    ) -> PerformanceMapResult:
        """
        Generate off-design performance maps.

        Args:
            engine_result: Completed engine design containing geometry and
                design-point performance (``dt_mm``, ``expansion_ratio``,
                ``thrust_vac``, ``pc_bar``, etc.).
            combustion: Combustion result providing ``gamma``, ``cstar``,
                and ``isp_vac``.
            config: Sweep ranges and resolution settings.

        Returns:
            A frozen ``PerformanceMapResult`` dataclass.
        """
        if not self.validate_inputs(engine_result, combustion, config):
            raise ValueError("PerformanceMapSolver: invalid inputs (see log for details)")

        # ----- Derived quantities from design point -----
        dt_m = engine_result.dt_mm * 1e-3
        At_m2 = math.pi / 4.0 * dt_m**2
        eps = engine_result.expansion_ratio
        gamma = combustion.gamma
        cstar = combustion.cstar
        pc_pa = engine_result.pc_bar * 1e5

        # Vacuum thrust coefficient: Cf_vac = F_vac / (Pc * At)
        cf_vac = (
            engine_result.thrust_vac / (pc_pa * At_m2) if (pc_pa * At_m2) > 0 else 0.0
        )

        logger.info(
            "PerformanceMapSolver: dt=%.2f mm, eps=%.1f, Cf_vac=%.4f, Pc=%.1f bar",
            engine_result.dt_mm,
            eps,
            cf_vac,
            engine_result.pc_bar,
        )

        # ----- 1. Altitude performance curve -----
        altitudes = np.linspace(
            config.altitude_range_m[0],
            config.altitude_range_m[1],
            config.altitude_points,
        )
        alt_data = altitude_performance_curve(
            pc_pa, At_m2, eps, gamma, cstar, cf_vac, altitudes
        )

        # ----- 2. Throttle map (linear Pc scaling) -----
        throttle_pcts = np.linspace(
            config.throttle_range[0],
            config.throttle_range[1],
            config.throttle_points,
        )
        pc_throttle = np.empty_like(throttle_pcts)
        thrust_throttle = np.empty_like(throttle_pcts)
        isp_throttle = np.empty_like(throttle_pcts)

        for i, thr in enumerate(throttle_pcts):
            pc_i = pc_pa * thr
            # Cf_vac is geometry-dependent and does not change with Pc for
            # ideal gas / frozen-composition assumptions.
            sea_level = thrust_at_altitude(
                pc_i, At_m2, eps, gamma, 101325.0, cstar, cf_vac
            )
            pc_throttle[i] = pc_i * 1e-5  # store in bar
            thrust_throttle[i] = sea_level["thrust_n"]
            isp_throttle[i] = sea_level["isp_s"]

        # ----- 3. MR sweep (placeholder -- needs combustion solver) -----
        # A proper MR sweep requires re-running CEA at each MR, which
        # couples this solver to CombustionSolver.  For now, leave the
        # MR fields as None and log a note.
        logger.info("MR sweep skipped (requires combustion solver integration)")

        return PerformanceMapResult(
            # Altitude
            altitudes_m=alt_data["altitudes"],
            thrust_vs_alt=alt_data["thrust"],
            isp_vs_alt=alt_data["isp"],
            cf_vs_alt=alt_data["cf"],
            # Throttle
            throttle_pcts=throttle_pcts,
            pc_vs_throttle=pc_throttle,
            thrust_vs_throttle=thrust_throttle,
            isp_vs_throttle=isp_throttle,
            # MR (not yet implemented)
            mr_values=None,
            isp_vs_mr=None,
            cstar_vs_mr=None,
        )
